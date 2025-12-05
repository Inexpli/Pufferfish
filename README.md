# Pufferfish ♟️
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A hybrid chess engine combining the minimax algorithm with alpha-beta pruning, supported by a neural network for best move prediction.

<br>

<img width="400" height="400" alt="image" src="https://images.chesscomfiles.com/uploads/game-gifs/90px/green/neo/0/cc/0/0/95bfafb4b6200460b1b095f7f13c1d6ec7859187d17a55ba38af7e9e3ace1528.gif" />


## 🎯 Features

- **Hybrid Architecture**: Minimax algorithm with alpha-beta pruning assisted by a neural network
- **Advanced Search Techniques**: Quiescence search and move ordering for better performance
- **Transposition Table**: Storage of calculated positions for faster move calculation
- **Position Evaluation**: Piece value heuristics and positional values for each piece
- **Neural Network**: PyTorch model predicting the best moves based on the position
- **UCI Compliance**: Full implementation of the Universal Chess Interface protocol
- **Tablebases**: Support for Gaviota tablebases
- **Opening Books**: Integration with Polyglot opening books
- **Cross-platform**: Runs on Windows, Linux, and macOS

## 📋 Requirements

- Python 3.8 or newer
- PyTorch
- Additional dependencies listed in `requirements.txt`

## 🚀 Installation

### Step 1: Clone the repository

```bash
git clone [https://github.com/Inexpli/Pufferfish.git](https://github.com/Inexpli/Pufferfish.git)
cd Pufferfish/
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Build the executable

To create a standalone executable engine:

```bash
pyinstaller pufferfish.spec
```

Note: You may expect a large number of warnings during the executable creation; do not worry, as this is normal for PyInstaller.

### Step 4: Export the engine

After compilation is complete, you will find the executable engine in the `dist/uci_wrapper/` folder.

## 🎮 Usage

### UCI Mode (with Graphical Interface)

Pufferfish can be used with any GUI supporting the UCI protocol, such as:
- Arena Chess GUI
- Cute Chess
- ChessBase
- Lichess (via Lichess-Bot)
- Chess.com

In the GUI settings, add the engine pointing to:
- **Source file:**: `pufferfish.py` (Python)
- **Executable**: `dist/pufferfish/pufferfish.exe` (Windows) or `dist/pufferfish/pufferfish` (Linux/Mac)

### CLI Mode (Command Line)

```bash
python pufferfish.py
```

Basic UCI commands:
```
uci                # Engine information
isready            # Check readiness
ucinewgame         # New game
position startpos  # Start position
go movetime 3000   # Search for 3 seconds
quit               # Exit
```

## 📁 Project Structure
```
Pufferfish/
├── charts/                        # Data and charts from the training process
│   ├── policy_network/
│   |   ├── [model_name].csv       # Metrics for each model (loss, accuracy, etc.)
│   |   └── read_chart.ipynb       # Jupyter notebook for reading data
|   └── value_network/
|       └── [model_name].csv       # Metric for the model (loss, accuracy, etc.)
├── core/                          
│   ├── evaluation.py              # Position evaluation functions
│   ├── minimax.py                 # Minimax algorithm with alpha-beta pruning and QS
│   ├── transposition_table.py     # Transposition table for search optimization
│   ├── heuristics.py              # Evaluation heuristics (material, position, etc.)
│   ├── model.py                   # Integration of ML models with the engine
│   ├── gaviota.py                 # Handling Gaviota tablebases
│   ├── polyglot.py                # Handling Polyglot opening books
│   ├── syzygy.py                  # Handling Syzygy tablebases
│   └── utils.py                   # Utility functions
├── models/
│   ├── policy_network/
│   |   └── [model_name].onnx      # Neural network model for move prediction
|   └── value_network/
|       └── [model_name].pth       # Neural network model for position evaluation
├── tablebases/
│   ├── gaviota/                   # Gaviota tablebases
│   └── polyglot/                  # Polyglot opening books
├── tests/                         
│   ├── methods.ipynb              # Performance tests of various minimax implementations
│   ├── minimax_opt.ipynb          # Minimax algorithm optimization
│   ├── nodes.ipynb                # Analysis of searched nodes
│   ├── gaviota.ipynb              # Integration tests with Gaviota bases
│   ├── polyglot.ipynb             # Integration tests with opening books
│   └── syzygy.ipynb               # Integration tests with Syzygy bases
├── training/                      
│   ├── policy_network/            # Training policy network (move prediction)
│   │   ├── data_manager.py        # Training data management
|   |   ├── data_parser.ipynb      # Processing PGN files for model learning
│   │   ├── dataset.py             # Policy network dataset
|   |   ├── lmdb_dataset.py        # Database configuration for games
│   │   ├── model.py               # Policy network architecture
│   │   ├── test_model.ipynb       # Policy model tests
│   │   └── train_model.ipynb      # Policy network training notebook
│   └── value_network/             # Training value network (position evaluation)
│       ├── data_manager.py        # Training data management
│       ├── dataset.py             # Value network dataset
│       ├── model.py               # Value network architecture
│       ├── test_minimax.ipynb     # Value network integration tests with minimax
│       ├── test_model.ipynb       # Value model tests
│       └── train_model.ipynb      # Value network training notebook
├── .gitignore                     # Files ignored by Git
├── README.md                      # You are reading this file
├── export_onnx.ipynb              # PyTorch model conversion to ONNX
├── pufferfish.py                  # Main UCI file
├── pufferfish.spec                # Specification for PyInstaller
└── requirements.txt               # Python dependencies
```

### Key Components

- **engine.py** - The heart of the project combining traditional chess techniques (minimax, alpha-beta pruning, transposition tables) with neural network predictions, implementing a hybrid approach to position evaluation.
- **core/** - Engine logic modules using the python-chess library for game representation, with custom implementations of search algorithms, evaluation, and integration with opening and endgame databases.
- **training/** - Complete pipeline for training two types of networks: policy network (predicting best moves) and value network (position evaluation).
- **charts/** - Training process visualizations enabling convergence monitoring and issue identification.
- **tests/** - Notebooks with optimization experiments, performance tests of various implementations, and integration with chess databases.

## 🧠 Architecture

Pufferfish uses a hybrid approach combining:

### Search Algorithm
1. **Minimax with alpha-beta pruning** - efficient game tree search with elimination of irrelevant branches.
2. **Quiescence search** - additional search in "unquiet" positions (captures, checks).
3. **Move ordering** - intelligent move analysis order for better pruning.
4. **Transposition table** - cache of calculated positions for faster recalculation of repeating positions.

### Position Evaluation
- **Piece value heuristics** - material valuation (pawn=100, knight=320, bishop=330, rook=500, queen=900).
- **Positional values** - bonuses/penalties for the position of each piece on the board (piece-square tables).

### Neural Network
- **PyTorch model** trained to predict the best moves
- **Input**: Representation of the current board position
- **Output**: Probabilities for possible moves
- **Integration**: The network assists the classic algorithm in selecting the best variations

### Databases
- **Gaviota tablebases** - optimal play in endgames (up to 5 pieces)
- **Polyglot opening books** - proven opening variations

## ⚙️ Configuration

The engine can be configured via standard UCI options. Available parameters depend on implementation and can be set in the GUI or via the `setoption` command.

## 🤝 Contributing

Contribution to the project's development is welcome! If you want to help:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/FeatureName`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/FeatureName`)
5. Open a Pull Request

## 📝 License

The entire project is under the MIT license.

## 📧 Contact

- GitHub: [@Inexpli](https://github.com/Inexpli)
- Repository: [https://github.com/Inexpli/Pufferfish](https://github.com/Inexpli/Pufferfish)

## 🙏 Acknowledgments

- Internet creators and YouTubers for inspiration and publishing materials regarding chess engine architecture
- The chess programming community for documentation and advice
- The PyTorch team for the deep learning framework

---

**Note**: Some features may change in the future.
