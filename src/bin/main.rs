use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use idz::{IdentityDisk, models::{QueryVector, Chunk, SearchResult}}; // Updated idz imports, removed DiskError
use ratatui::{
    backend::{CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use std::fs;
use std::io;
use std::path::PathBuf;
// use std::ops::Deref; // No longer needed

#[derive(Parser)]
#[command(name = "idz-cli")]
#[command(about = "A TUI for manipulating Identity Disk (.idz) files")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new .idz file from text files
    Create {
        /// Output .idz file path
        #[arg(short, long)]
        output: PathBuf,
        /// Text files to process
        files: Vec<PathBuf>,
        /// Embedding model signature (e.g., "openai/text-embedding-ada-002_fp32")
        #[arg(short, long, default_value = "openai/text-embedding-ada-002_fp32")]
        model_signature: String,
    },
    /// Explore an existing .idz file with TUI
    Explore {
        /// .idz file to explore
        file: PathBuf,
        /// Model signature to load for searching (e.g., "openai/text-embedding-ada-002_fp32")
        #[arg(short, long)]
        model_signature: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Create { output, files, model_signature } => {
            create_idz_file(output, files, &model_signature)?;
        }
        Commands::Explore { file, model_signature } => {
            run_tui(file, &model_signature)?;
        }
    }

    Ok(())
}

fn create_idz_file(output: PathBuf, files: Vec<PathBuf>, model_signature: &str) -> Result<()> {
    println!("Creating .idz file: {:?}", output);
    println!("Model Signature: {}", model_signature);

    // Determine embedding dimension from model_signature (very basic parsing)
    // E.g., "model-name-1536_fp32" -> 1536. This is a simplification.
    // A more robust solution would involve a lookup or more structured signature.
    let dim: usize = model_signature.split('_').next().unwrap_or("").split('-').last()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1536); // Default if parsing fails

    // Check if the model signature implies f32, otherwise this dummy generation is wrong
    if !model_signature.contains("fp32") && model_signature.contains('_') {
        // if there's a type specified and it's not fp32
        eprintln!("Warning: Model signature '{}' does not explicitly state 'fp32'. Dummy f32 embeddings will be generated. This might be incorrect.", model_signature);
    }


    let mut disk = IdentityDisk::create(&output, model_signature)?;

    for file_path in files {
        println!("Processing file: {:?}", file_path);
        let content = fs::read_to_string(&file_path)?;
        
        // Split content into chunks (simple line-based chunking for demo)
        let chunks: Vec<&str> = content.lines().filter(|line| !line.trim().is_empty()).collect();
        
        for (i, chunk_content) in chunks.iter().enumerate() {
            let meta = serde_json::json!({
                "source_file": file_path.to_string_lossy(),
                "chunk_index": i,
                "char_count": chunk_content.len()
            });
            
            // Generate dummy f32 embedding
            let embedding_values: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            let query_vector = QueryVector::F32(&embedding_values);
            
            match disk.add_chunk(chunk_content, query_vector, Some(meta)) {
                Ok(chunk_id) => println!("Added chunk {} from {:?}", chunk_id, file_path),
                Err(e) => eprintln!("Failed to add chunk from {:?}: {}", file_path, e),
            }
        }
    }

    println!("Successfully created .idz file at {:?}!", output);
    Ok(())
}

// Simple random number generator for demo embeddings
mod rand {
    static mut SEED: u32 = 1;
    
    pub fn random<T>() -> T 
    where 
        T: From<f32>,
    {
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            let val = (SEED >> 16) as f32 / 65536.0; // Always positive 0-1
            T::from(val)
        }
    }
}

fn run_tui(file_path: PathBuf, model_signature: &str) -> Result<()> {
    // Load the .idz file
    println!("Opening .idz file: {:?} with model_signature: {}", file_path, model_signature);
    let disk = IdentityDisk::open(&file_path, model_signature)?;
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let app = App::new(disk, file_path, model_signature.to_string());
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

struct App {
    disk: IdentityDisk, // This is now the new IdentityDisk
    file_path: PathBuf,
    model_signature: String, // Store the model signature used to open the disk
    all_chunks: Vec<Chunk>, // Cache all chunks
    current_view: AppView,
    list_state: ListState, // For navigating all_chunks
    search_list_state: ListState, // For navigating search_results
    selected_chunk_id: Option<String>, // Store ID of the selected chunk for detail view
    // OR: selected_chunk_idx: Option<usize> to index into all_chunks / search_results.chunks
    search_mode: bool,
    search_query: String,
    search_results: Vec<SearchResult>, // Stores SearchResult structs
    status_message: String, // For displaying errors or info
}

#[derive(PartialEq)]
enum AppView {
    Overview,
    ChunkList,
    ChunkDetail,
    Search,
}

impl App {
    fn new(disk: IdentityDisk, file_path: PathBuf, model_signature: String) -> Self {
        let list_state = ListState::default(); // Removed mut
        let mut app = Self {
            disk,
            file_path,
            model_signature,
            all_chunks: Vec::new(), // Will be loaded by refresh_chunks
            current_view: AppView::Overview,
            list_state,
            search_list_state: ListState::default(),
            selected_chunk_id: None,
            search_mode: false,
            search_query: String::new(),
            search_results: Vec::new(),
            status_message: String::new(),
        };
        app.refresh_chunks(); // Load initial chunks
        if !app.all_chunks.is_empty() {
            app.list_state.select(Some(0));
        }
        app
    }

    fn refresh_chunks(&mut self) {
        match self.disk.get_chunks() {
            Ok(chunks) => {
                self.all_chunks = chunks;
                if self.all_chunks.is_empty() {
                    self.list_state.select(None);
                    self.status_message = "No chunks found in the disk.".to_string();
                } else {
                     // Try to keep selection if possible, otherwise select first
                    let selected = self.list_state.selected();
                    if selected.is_none() || selected.unwrap_or(0) >= self.all_chunks.len() {
                        self.list_state.select(Some(0));
                    }
                }
            }
            Err(e) => {
                self.all_chunks.clear();
                self.list_state.select(None);
                self.status_message = format!("Error loading chunks: {}", e);
            }
        }
    }
    
    fn next_chunk(&mut self) {
        let count = self.all_chunks.len();
        if count == 0 { return; }
        let i = match self.list_state.selected() {
            Some(i) => if i >= count - 1 { 0 } else { i + 1 },
            None => 0,
        };
        self.list_state.select(Some(i));
    }

    fn previous_chunk(&mut self) {
        let count = self.all_chunks.len();
        if count == 0 { return; }
        let i = match self.list_state.selected() {
            Some(i) => if i == 0 { count - 1 } else { i - 1 },
            None => 0,
        };
        self.list_state.select(Some(i));
    }

    fn next_search_result(&mut self) {
        let count = self.search_results.len();
        if count == 0 { return; }
        let i = match self.search_list_state.selected() {
            Some(i) => if i >= count - 1 { 0 } else { i + 1 },
            None => 0,
        };
        self.search_list_state.select(Some(i));
    }

    fn previous_search_result(&mut self) {
        let count = self.search_results.len();
        if count == 0 { return; }
        let i = match self.search_list_state.selected() {
            Some(i) => if i == 0 { count - 1 } else { i - 1 },
            None => 0,
        };
        self.search_list_state.select(Some(i));
    }

    // fn get_chunk_count(&self) -> usize { // Replaced by self.all_chunks.len()
    //     self.all_chunks.len()
    // }

    fn perform_search(&mut self) {
        if self.search_query.is_empty() {
            self.search_results.clear();
            self.search_list_state.select(None);
            return;
        }
        
        // Use model_signature to get dim, similar to create_idz_file
        // This is a simplification. A robust app might store dim or parse more reliably.
        let dim: usize = self.model_signature.split('_').next().unwrap_or("").split('-').last()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1536); // Default if parsing fails

        // Generate dummy f32 embedding for the search query
        // Ensure this matches the expected QueryVector type for the loaded index.
        // For now, assumes F32 based on common use and previous dummy data.
        let query_embedding_values: Vec<f32> = (0..dim).map(|_| rand::random::<f32>() * 0.1).collect(); // small values
        let query_vec = QueryVector::F32(&query_embedding_values);
        
        match self.disk.search(query_vec, 10) {
            Ok(results) => {
                self.search_results = results;
                if !self.search_results.is_empty() {
                    self.search_list_state.select(Some(0));
                } else {
                    self.search_list_state.select(None);
                }
                self.status_message = format!("Found {} results for '{}'", self.search_results.len(), self.search_query);
            }
            Err(e) => {
                self.search_results.clear();
                self.search_list_state.select(None);
                self.status_message = format!("Search error: {}", e);
            }
        }
        
        // Select the first result if any
        if !self.search_results.is_empty() {
            self.search_list_state.select(Some(0));
        }
    }
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, mut app: App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                // Global keybindings
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('1') => app.current_view = AppView::Overview,
                    KeyCode::Char('2') => app.current_view = AppView::ChunkList,
                    KeyCode::Char('3') => app.current_view = AppView::Search,
                    _ => {}
                }

                // View-specific keybindings
                if app.search_mode {
                     match key.code {
                        KeyCode::Enter => {
                            app.search_mode = false;
                            app.perform_search();
                        }
                        KeyCode::Char(c) => app.search_query.push(c),
                        KeyCode::Backspace => { app.search_query.pop(); },
                        KeyCode::Esc => {
                            app.search_mode = false;
                            app.search_query.clear();
                        }
                        _ => {}
                    }
                } else {
                    match app.current_view {
                        AppView::ChunkList => match key.code {
                            KeyCode::Down | KeyCode::Char('j') => app.next_chunk(),
                            KeyCode::Up | KeyCode::Char('k') => app.previous_chunk(),
                            KeyCode::Enter => {
                                if let Some(selected_idx) = app.list_state.selected() {
                                    if let Some(chunk) = app.all_chunks.get(selected_idx) {
                                        app.selected_chunk_id = Some(chunk.chunk_id.clone());
                                        app.current_view = AppView::ChunkDetail;
                                    }
                                }
                            }
                            _ => {}
                        },
                        AppView::ChunkDetail => match key.code {
                            KeyCode::Esc => app.current_view = AppView::ChunkList,
                            _ => {}
                        },
                        AppView::Search => match key.code {
                            KeyCode::Char('/') => {
                                app.search_mode = true;
                                app.search_results.clear(); // Clear old results
                            }
                            KeyCode::Down | KeyCode::Char('j') => app.next_search_result(),
                            KeyCode::Up | KeyCode::Char('k') => app.previous_search_result(),
                            KeyCode::Enter => {
                                if let Some(selected_idx) = app.search_list_state.selected() {
                                    if let Some(search_result) = app.search_results.get(selected_idx) {
                                        app.selected_chunk_id = Some(search_result.chunk.chunk_id.clone());
                                        app.current_view = AppView::ChunkDetail;
                                    }
                                }
                            }
                            _ => {}
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(f.size());

    // Header
    let header = Paragraph::new(format!("IDZ Explorer - {}", app.file_path.display()))
        .block(Block::default().borders(Borders::ALL).title("Identity Disk Explorer"));
    f.render_widget(header, chunks[0]);

    // Footer with controls
    let footer_text = if app.search_mode {
        "Enter: Search | Esc: Cancel"
    } else {
        match app.current_view {
            AppView::Overview => "1: Overview | 2: Chunks | 3: Search | q: Quit",
            AppView::ChunkList => "↑↓/jk: Navigate | Enter: View | 1: Overview | 3: Search | q: Quit",
            AppView::ChunkDetail => "Esc: Back | 1: Overview | 2: Chunks | q: Quit",
            AppView::Search => "/: Search | ↑↓/jk: Navigate Results | Enter: View Chunk | q: Quit",
        }
    };
    let footer = Paragraph::new(footer_text)
        .block(Block::default().borders(Borders::ALL).title("Controls"));
    f.render_widget(footer, chunks[2]);

    // Main content
    match app.current_view {
        AppView::Overview => render_overview(f, chunks[1], app),
        AppView::ChunkList => render_chunk_list(f, chunks[1], app),
        AppView::ChunkDetail => render_chunk_detail(f, chunks[1], app),
        AppView::Search => render_search(f, chunks[1], app),
    }
}

fn render_overview(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // File info
    let spec_version = app.disk.get_spec_version().unwrap_or_else(|e| format!("Error: {}", e));

    let total_chars: usize = app.all_chunks.iter().map(|c| c.content.len()).sum();
    let avg_chars = if !app.all_chunks.is_empty() {
        total_chars / app.all_chunks.len()
    } else {
        0
    };

    let file_info = vec![
        format!("File: {}", app.file_path.display()),
        format!("Spec Version: {}", spec_version),
        format!("Model Signature: {}", app.model_signature),
        format!("Total Chunks: {}", app.all_chunks.len()),
        format!("Average chars per chunk: {}", avg_chars),
    ];
    
    let file_widget = Paragraph::new(file_info.join("\n"))
        .block(Block::default().borders(Borders::ALL).title("Disk Information"))
        .wrap(Wrap { trim: true });
    f.render_widget(file_widget, chunks[0]); // Use the full area for simplified overview

    // Embedding info (simplified or extracted from model_signature)
    let dim: usize = app.model_signature.split('_').next().unwrap_or("").split('-').last()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // Show 0 if not parsable
    let dtype = app.model_signature.split('_').nth(1).unwrap_or("unknown");

    let index_type_desc = app.disk.get_index_type_description().unwrap_or_else(|e| format!("Error: {}", e));
    let embed_info = vec![
        format!("Parsed Dimension: {}", if dim == 0 { "N/A".to_string() } else { dim.to_string() }),
        format!("Parsed Data Type: {}", dtype),
        format!("Active Index Type: {}", index_type_desc),
    ];
    let embed_widget = Paragraph::new(embed_info.join("\n"))
        .block(Block::default().borders(Borders::ALL).title("Active Index Information"))
        .wrap(Wrap { trim: true });
    f.render_widget(embed_widget, chunks[1]);
}

fn render_chunk_list(f: &mut Frame, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = app.all_chunks.iter().enumerate()
        .map(|(_i, chunk)| { // _i instead of i
            let preview = if chunk.content.len() > 80 {
                format!("{}...", chunk.content.chars().take(77).collect::<String>())
            } else {
                chunk.content.clone()
            };
            ListItem::new(format!("ID: {}... | {}", &chunk.chunk_id[..8], preview))
        })
        .collect();

    let list_title = format!("Text Chunks (Total: {})", app.all_chunks.len());
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(list_title))
        .highlight_style(Style::default().bg(Color::Blue).fg(Color::White))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn render_chunk_detail(f: &mut Frame, area: Rect, app: &App) {
    if let Some(ref selected_id) = app.selected_chunk_id {
        if let Some(chunk) = app.all_chunks.iter().find(|c| &c.chunk_id == selected_id) {
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                .split(area);

            let text_widget = Paragraph::new(chunk.content.clone())
                .block(Block::default().borders(Borders::ALL).title(format!("Chunk {} - Text", selected_id)))
                .wrap(Wrap { trim: true });
            f.render_widget(text_widget, layout[0]);

            let meta_text = serde_json::to_string_pretty(&chunk.metadata)
                .unwrap_or_else(|_| "Invalid JSON".to_string());
            let meta_widget = Paragraph::new(meta_text)
                .block(Block::default().borders(Borders::ALL).title("Metadata"))
                .wrap(Wrap { trim: true });
            f.render_widget(meta_widget, layout[1]);
        } else {
            let error_widget = Paragraph::new(format!("Could not find chunk with ID: {}", selected_id))
                .block(Block::default().borders(Borders::ALL).title("Error"));
            f.render_widget(error_widget, area);
        }
    } else {
         let info_widget = Paragraph::new("No chunk selected.")
            .block(Block::default().borders(Borders::ALL).title("Chunk Detail"));
        f.render_widget(info_widget, area);
    }
}

fn render_search(f: &mut Frame, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    // Search input
    let search_style = if app.search_mode {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };
    
    // Add a blinking cursor effect
    let query_display = if app.search_mode {
        format!("Search: {}|", app.search_query)
    } else {
        format!("Search: {}", app.search_query)
    };

    let search_widget = Paragraph::new(query_display)
        .block(Block::default().borders(Borders::ALL).title("Semantic Search").border_style(search_style));
    f.render_widget(search_widget, chunks[0]);

    // Search results
    if !app.search_results.is_empty() {
        let items: Vec<ListItem> = app.search_results.iter().map(|result| {
            let chunk = &result.chunk;
            let score = result.distance;
            let preview = if chunk.content.len() > 60 { // Adjusted length for more info
                format!("{}...", chunk.content.chars().take(57).collect::<String>())
            } else {
                chunk.content.clone()
            };
            ListItem::new(format!("ID: {}... | Score: {:.4} | {}", &chunk.chunk_id[..8], score, preview))
        }).collect();

        let list_title = format!("Search Results (Found: {})", app.search_results.len());
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(list_title))
            .highlight_style(Style::default().bg(Color::Blue).fg(Color::White))
            .highlight_symbol("> ");

        f.render_stateful_widget(list, chunks[1], &mut app.search_list_state);
    } else {
        let results_text = if app.search_query.is_empty() {
            "Press '/' to start searching. Press Enter to perform search."
        } else {
            "No results found."
        };
        
        let results_widget = Paragraph::new(results_text)
            .block(Block::default().borders(Borders::ALL).title("Search Results"))
            .wrap(Wrap { trim: true });
        f.render_widget(results_widget, chunks[1]);
    }
}