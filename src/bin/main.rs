use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use idz::{IdentityDisk, IdentityDiskBuilder};
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
        /// Embedding model name
        #[arg(short, long, default_value = "text-embedding-ada-002")]
        model: String,
        /// Embedding dimension
        #[arg(short, long, default_value = "1536")]
        dim: usize,
    },
    /// Explore an existing .idz file with TUI
    Explore {
        /// .idz file to explore
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Create { output, files, model, dim } => {
            create_idz_file(output, files, model, dim)?;
        }
        Commands::Explore { file } => {
            run_tui(file)?;
        }
    }

    Ok(())
}

fn create_idz_file(output: PathBuf, files: Vec<PathBuf>, model: String, dim: usize) -> Result<()> {
    println!("Creating .idz file: {:?}", output);
    println!("Model: {}, Dimension: {}", model, dim);

    let mut builder = IdentityDiskBuilder::new(dim, &model);

    for file_path in files {
        println!("Reading file: {:?}", file_path);
        let content = fs::read_to_string(&file_path)?;
        
        // Split content into chunks (simple line-based chunking for demo)
        let chunks: Vec<&str> = content.lines().filter(|line| !line.trim().is_empty()).collect();
        
        for (i, chunk) in chunks.iter().enumerate() {
            let meta = serde_json::json!({
                "source_file": file_path.to_string_lossy(),
                "chunk_index": i,
                "char_count": chunk.len()
            });
            
            // Generate dummy embedding (in real use case, you'd call an embedding API)
            let embedding: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            
            builder.push(chunk, meta, &embedding)?;
        }
    }

    builder.write(output)?;
    println!("Successfully created .idz file!");
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

fn run_tui(file_path: PathBuf) -> Result<()> {
    // Load the .idz file
    let disk = IdentityDisk::open(&file_path)?;
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let app = App::new(disk, file_path);
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
    disk: IdentityDisk,
    file_path: PathBuf,
    current_view: AppView,
    list_state: ListState,
    search_list_state: ListState, // Added for search results
    selected_chunk: Option<usize>,
    search_mode: bool,
    search_query: String,
    search_results: Vec<(usize, f32)>,
}

#[derive(PartialEq)]
enum AppView {
    Overview,
    ChunkList,
    ChunkDetail,
    Search,
}

impl App {
    fn new(disk: IdentityDisk, file_path: PathBuf) -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        
        Self {
            disk,
            file_path,
            current_view: AppView::Overview,
            list_state,
            search_list_state: ListState::default(), // Initialize search list state
            selected_chunk: None,
            search_mode: false,
            search_query: String::new(),
            search_results: Vec::new(),
        }
    }
    
    fn next_chunk(&mut self) {
        let count = self.get_chunk_count();
        if count == 0 { return; }
        let i = match self.list_state.selected() {
            Some(i) => if i >= count - 1 { 0 } else { i + 1 },
            None => 0,
        };
        self.list_state.select(Some(i));
    }

    fn previous_chunk(&mut self) {
        let count = self.get_chunk_count();
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
    
    fn get_chunk_count(&self) -> usize {
        self.disk.manifest.chunks.count
    }

    fn perform_search(&mut self) {
        if self.search_query.is_empty() {
            self.search_results.clear();
            return;
        }
        
        // Generate a dummy query embedding, just like in create_idz_file
        let dim = self.disk.manifest.embedding.dim;
        let query_embedding: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        
        // Perform the search (we'll ask for the top 10 results)
        self.search_results = self.disk.search(&query_embedding, 10);
        
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
                                app.selected_chunk = app.list_state.selected();
                                app.current_view = AppView::ChunkDetail;
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
                                    if let Some((chunk_id, _)) = app.search_results.get(selected_idx) {
                                        app.selected_chunk = Some(*chunk_id);
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
    let manifest = &app.disk.manifest;
    let file_info = vec![
        format!("File: {}", app.file_path.display()),
        format!("Version: {}", manifest.version),
        format!("Created: {}", manifest.created),
        format!("Chunks: {}", manifest.chunks.count),
        format!("Average chars per chunk: {}", manifest.chunks.average_chars),
    ];
    
    let file_widget = Paragraph::new(file_info.join("\n"))
        .block(Block::default().borders(Borders::ALL).title("File Information"))
        .wrap(Wrap { trim: true });
    f.render_widget(file_widget, chunks[0]);

    // Embedding info
    let embed_info = vec![
        format!("Model: {}", manifest.embedding.model),
        format!("Dimensions: {}", manifest.embedding.dim),
        format!("Data type: {}", manifest.embedding.dtype),
        format!("Quantized: {}", manifest.embedding.quantised),
        format!("Index: {} (M={}, ef_construct={})", 
               manifest.index.kind, manifest.index.m, manifest.index.ef_construct),
    ];
    
    let embed_widget = Paragraph::new(embed_info.join("\n"))
        .block(Block::default().borders(Borders::ALL).title("Embedding & Index Information"))
        .wrap(Wrap { trim: true });
    f.render_widget(embed_widget, chunks[1]);
}

fn render_chunk_list(f: &mut Frame, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = (0..app.get_chunk_count())
        .map(|i| {
            let text = app.disk.chunk_text(i);
            let preview = if text.len() > 80 {
                format!("{}...", &text[..77])
            } else {
                text.to_string()
            };
            ListItem::new(format!("{}: {}", i, preview))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Text Chunks"))
        .highlight_style(Style::default().bg(Color::Blue).fg(Color::White))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn render_chunk_detail(f: &mut Frame, area: Rect, app: &App) {
    if let Some(chunk_id) = app.selected_chunk {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        // Text content
        let text = app.disk.chunk_text(chunk_id);
        let text_widget = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title(format!("Chunk {} - Text", chunk_id)))
            .wrap(Wrap { trim: true });
        f.render_widget(text_widget, chunks[0]);

        // Metadata
        let meta = app.disk.chunk_meta(chunk_id);
        let meta_text = serde_json::to_string_pretty(meta).unwrap_or_else(|_| "Invalid JSON".to_string());
        let meta_widget = Paragraph::new(meta_text)
            .block(Block::default().borders(Borders::ALL).title("Metadata"))
            .wrap(Wrap { trim: true });
        f.render_widget(meta_widget, chunks[1]);
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
        let items: Vec<ListItem> = app.search_results.iter().map(|(id, score)| {
            let text = app.disk.chunk_text(*id);
            let preview = if text.len() > 80 {
                format!("{}...", text.chars().take(77).collect::<String>())
            } else {
                text.to_string()
            };
            ListItem::new(format!("ID: {:<4} | Score: {:.4} | {}", id, score, preview))
        }).collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Search Results"))
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