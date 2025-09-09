#include <iostream>
#include <limits>
#include <omp.h>
#include "chess.hpp"

using namespace chess;
using namespace std;

constexpr int ALL_PIECES =
    PieceGenType::PAWN | PieceGenType::KNIGHT | PieceGenType::BISHOP |
    PieceGenType::ROOK | PieceGenType::QUEEN | PieceGenType::KING;

int evaluate(const Board &board) {
    int score = 0;
    for (int i = 0; i < 64; i++) {
        Square sq = static_cast<Square>(i);
        Piece piece = board.at(sq);
        if (piece == Piece::NONE) continue;

        int val = 0;
        switch (piece.type().internal()) {
            case PieceType::PAWN:   val = 100; break;
            case PieceType::KNIGHT: val = 320; break;
            case PieceType::BISHOP: val = 330; break;
            case PieceType::ROOK:   val = 500; break;
            case PieceType::QUEEN:  val = 900; break;
            case PieceType::KING:   val = 20000; break;
            default: break;
        }
        if (piece.color() == Color::WHITE) score += val;
        else score -= val;
    }
    return score;
}

int minimax(Board &board, int depth, int alpha, int beta, bool maximizingPlayer) {
    if (depth == 0) return evaluate(board);

    Movelist moves;
    movegen::legalmoves<movegen::MoveGenType::ALL>(moves, board, ALL_PIECES);

    if (moves.empty()) {
        if (board.inCheck()) return maximizingPlayer ? -1000000 : 1000000;
        return 0;
    }

    if (maximizingPlayer) {
        int maxEval = std::numeric_limits<int>::min();
        for (const Move &move : moves) {
            board.makeMove(move);
            int eval = minimax(board, depth - 1, alpha, beta, false);
            board.unmakeMove(move);
            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        int minEval = std::numeric_limits<int>::max();
        for (const Move &move : moves) {
            board.makeMove(move);
            int eval = minimax(board, depth - 1, alpha, beta, true);
            board.unmakeMove(move);
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) break;
        }
        return minEval;
    }
}

Move findBestMove(Board &board, int depth) {
    Movelist moves;
    movegen::legalmoves<movegen::MoveGenType::ALL>(moves, board, ALL_PIECES);

    Move bestMove = Move::NO_MOVE;
    int bestValue = std::numeric_limits<int>::min();

    // OpenMP równoległość
    #pragma omp parallel
    {
        Move localBestMove = Move::NO_MOVE;
        int localBestValue = std::numeric_limits<int>::min();

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < moves.size(); i++) {
            Board copy = board;
            copy.makeMove(moves[i]);
            int eval = minimax(copy, depth - 1,
                               std::numeric_limits<int>::min(),
                               std::numeric_limits<int>::max(),
                               false);

            if (eval > localBestValue) {
                localBestValue = eval;
                localBestMove = moves[i];
            }
        }

        #pragma omp critical
        {
            if (localBestValue > bestValue) {
                bestValue = localBestValue;
                bestMove = localBestMove;
            }
        }
    }

    return bestMove;
}

int main() {
    Board board("rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq d6 0 1");

    Move best = findBestMove(board, 8);

    std::cout << "Najlepszy ruch: " << uci::moveToUci(best) << "\n";

    return 0;
}