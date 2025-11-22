# Pufferfish ♟️

Hybrydowy silnik szachowy łączący algorytm minimax z alpha-beta pruning ze wsparciem sieci neuronowej do predykcji najlepszych ruchów.

## 🎯 Cechy

- **Architektura hybrydowa**: Algorytm minimax z alpha-beta pruning wspomagany siecią neuronową
- **Zaawansowane techniki wyszukiwania**: Quiescence search i move ordering dla lepszej wydajności
- **Tabela transpozycji**: Przechowywanie obliczonych pozycji dla szybszego przeliczania ruchów
- **Ewaluacja pozycji**: Heurystyki wartości figur oraz wartości pozycyjnej dla każdej figury
- **Sieć neuronowa**: Model PyTorch przewidujący najlepsze ruchy na podstawie pozycji
- **Zgodność UCI**: Pełna implementacja protokołu Universal Chess Interface
- **Bazy końcówkowe**: Wsparcie dla Gaviota tablebases
- **Książki otwarć**: Integracja z Polyglot opening books
- **Cross-platform**: Działa na systemach Windows, Linux i macOS

## 📋 Wymagania

- Python 3.8 lub nowszy
- PyTorch
- Dodatkowe zależności wymienione w `requirements.txt`

## 🚀 Instalacja

### Krok 1: Sklonuj repozytorium

```bash
git clone https://github.com/Inexpli/Pufferfish.git
cd Pufferfish/
```

### Krok 2: Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### Krok 3: Zbuduj wykonywalny plik (opcjonalnie)

Aby utworzyć standalone wykonywalny silnik:

```bash
pyinstaller pufferfish.spec
```

**Uwaga**: Podczas tworzenia pliku wykonywalnego możesz spodziewać się sporej liczby ostrzeżeń, nie przejmuj się jednak gdyż to normalne w przypadku **PyInstaller'a**.

### Krok 4: Eksportuj silnik

Po zakończeniu kompilacji, wykonywalny silnik znajdziesz w folderze `dist/uci_wrapper/`.

## 🎮 Użycie

### Tryb UCI (z interfejsem graficznym)

Pufferfish można używać z dowolnym GUI wspierającym protokół UCI, takim jak:
- Arena Chess GUI
- Cute Chess
- ChessBase
- Lichess (poprzez Lichess-Bot)
- Chess.com

W ustawieniach GUI dodaj silnik wskazując na:
- **Plik źródłowy**: `pufferfish.py` (Python)
- **Wykonywalny**: `dist/pufferfish/pufferfish.exe` (Windows) lub `dist/pufferfish/pufferfish` (Linux/Mac)

### Tryb CLI (wiersz poleceń)

```bash
python pufferfish.py
```

Podstawowe komendy UCI:
```
uci                # Informacje o silniku
isready            # Sprawdzenie gotowości
ucinewgame         # Nowa partia
position startpos  # Pozycja startowa
go movetime 3000   # Szukaj przez 3 sekundy
quit               # Wyjście
```

## 📁 Struktura projektu
```
Pufferfish/
├── charts/                        # Dane i wykresy z procesu uczenia
│   ├── policy_network/
│   |   ├── [model_name].csv       # Metryki dla każdego modelu (loss, accuracy, itp.)
│   |   └── read_chart.ipynb       # Jupyter notebook do odczytania danych
|   └── value_network/
|       └── [model_name].csv       # Metryka dla modelu (loss, accuracy, itp.)
├── core/                          
│   ├── evaluation.py              # Funkcje ewaluacji pozycji
│   ├── minimax.py                 # Algorytm Minimax z alpha-beta pruning i QS
│   ├── transposition_table.py     # Tabela transpozycji dla optymalizacji przeszukiwania
│   ├── heuristics.py              # Heurystyki ewaluacji (materiał, pozycja, itp.)
│   ├── model.py                   # Integracja modeli ML z silnikiem
│   ├── gaviota.py                 # Obsługa baz końcówkowych Gaviota
│   ├── polyglot.py                # Obsługa opening books Polyglot
│   ├── syzygy.py                  # Obsługa baz końcówkowych Syzygy
│   └── utils.py                   # Funkcje pomocnicze
├── models/
│   ├── policy_network/
│   |   └── [model_name].onnx      # Model sieci neuronowej do predykcji ruchu
|   └── value_network/
|       └── [model_name].pth       # Model sieci neuronowej do oceny pozycji
├── tablebases/
│   ├── gaviota/                   # Bazy końcówkowe Gaviota
│   └── polyglot/                  # Opening books Polyglot
├── tests/                         
│   ├── methods.ipynb              # Testy wydajności różnych implementacji minimax
│   ├── minimax_opt.ipynb          # Optymalizacja algorytmu minimax
│   ├── nodes.ipynb                # Analiza przeszukiwanych węzłów
│   ├── gaviota.ipynb              # Testy integracji z bazami Gaviota
│   ├── polyglot.ipynb             # Testy integracji z opening books
│   └── syzygy.ipynb               # Testy integracji z bazami Syzygy
├── training/                      
│   ├── policy_network/            # Trening sieci policy (przewidywanie ruchów)
│   │   ├── data_manager.py        # Zarządzanie danymi treningowymi
|   |   ├── data_parser.ipynb      # Przetwarzanie plików PGN do nauki modelu
│   │   ├── dataset.py             # Dataset policy network
|   |   ├── lmdb_dataset.py        # Konfiguracja bazy danych dla partii
│   │   ├── model.py               # Architektura sieci policy
│   │   ├── test_model.ipynb       # Testy modelu policy
│   │   └── train_model.ipynb      # Notebook treningu policy network
│   └── value_network/             # Trening sieci value (ewaluacja pozycji)
│       ├── data_manager.py        # Zarządzanie danymi treningowymi
│       ├── dataset.py             # Dataset value network
│       ├── model.py               # Architektura sieci value
│       ├── test_minimax.ipynb     # Testy integracji value network z minimax
│       ├── test_model.ipynb       # Testy modelu value
│       └── train_model.ipynb      # Notebook treningu value network
├── .gitignore                     # Pliki ignorowane przez Git
├── README.md                      # Właśnie czytasz ten plik
├── export_onnx.ipynb              # Konwersja modelu PyTorch na ONNX
├── pufferfish.py                  # Główny plik UCI
├── pufferfish.spec                # Specyfikacja dla PyInstaller
└── requirements.txt               # Zależności Python
```

### Kluczowe komponenty

- **engine.py** - Serce projektu łączące tradycyjne techniki szachowe (minimax, alpha-beta pruning, tabele transpozycji) z predykcjami sieci neuronowej, implementujące hybrydowe podejście do ewaluacji pozycji
- **core/** - Moduły logiki silnika wykorzystujące bibliotekę python-chess do reprezentacji gry, z własnymi implementacjami algorytmów przeszukiwania, ewaluacji oraz integracją z bazami otwarć i końcówek
- **training/** - Kompletny pipeline do trenowania dwóch typów sieci: policy network (przewidywanie najlepszych ruchów) oraz value network (ewaluacja pozycji)
- **charts/** - Wizualizacje procesu uczenia umożliwiające monitorowanie konwergencji i identyfikację problemów
- **tests/** - Notebooki z eksperymentami optymalizacyjnymi, testami wydajności różnych implementacji oraz integracją z bazami danych szachowych

## 🧠 Architektura

Pufferfish wykorzystuje hybrydowe podejście łączące:

### Algorytm przeszukiwania
1. **Minimax z alpha-beta pruning** - efektywne przeszukiwanie drzewa gier z eliminacją nieistotnych gałęzi
2. **Quiescence search** - dodatkowe przeszukiwanie w "niespokojnych" pozycjach (bicia, szachy)
3. **Move ordering** - inteligentna kolejność analizy ruchów dla lepszego przycinania
4. **Tabela transpozycji** - cache obliczonych pozycji dla szybszego przeliczania powtarzających się pozycji

### Ewaluacja pozycji
- **Heurystyki wartości figur** - wycena materiału (pionek=100, skoczek=320 goniec=330, wieża=500, hetman=900)
- **Wartości pozycyjne** - bonusy/kary za pozycję każdej figury na planszy (piece-square tables)

### Sieć neuronowa
- **Model PyTorch** trenowany do predykcji najlepszych ruchów
- **Wejście**: Reprezentacja aktualnej pozycji na planszy
- **Wyjście**: Prawdopodobieństwa dla możliwych ruchów
- **Integracja**: Sieć wspomaga klasyczny algorytm w wyborze najlepszych wariantów

### Bazy danych
- **Gaviota tablebases** - optymalna gra w końcówkach (do 5 figur)
- **Polyglot opening books** - sprawdzone warianty otwarć

## ⚙️ Konfiguracja

Silnik można skonfigurować poprzez standardowe opcje UCI. Dostępne parametry zależą od implementacji i mogą być ustawione w GUI lub poprzez komendę `setoption`.

## 🤝 Wkład w rozwój

Wkład w rozwój projektu jest mile widziany! Jeśli chcesz pomóc:

1. Fork repozytorium
2. Stwórz branch dla swojej funkcjonalności (`git checkout -b feature/NazwaFunkcjonalnosci`)
3. Commituj zmiany (`git commit -m 'Dodaj nową funkcjonalność'`)
4. Push do brancha (`git push origin feature/NazwaFunkcjonalnosci`)
5. Otwórz Pull Request

## 📝 Licencja

Cały projekt jest na licencji MIT.

## 📧 Kontakt

- GitHub: [@Inexpli](https://github.com/Inexpli)
- Repozytorium: [https://github.com/Inexpli/Pufferfish](https://github.com/Inexpli/Pufferfish)

## 🙏 Podziękowania

- Twórcom internetowym oraz youtuberom za inspiracje oraz publikacje materiałów odnośnie architektury silników szachowych
- Społeczność chess programming za dokumentację oraz rady
- PyTorch team za framework do deep learning

---

**Uwaga**: W przyszłości niektóre funkcje mogą ulec zmianie.
