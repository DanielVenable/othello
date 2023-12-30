export const EMPTY = 0, WHITE = -1, BLACK = 1;

export class Game {
    validMoves = new Array(64).fill(0);

    constructor() {
        this.reset();
    }

    reset() {
        this.board = [];
        this.turn = BLACK;
        this.isDone = false;

        // fill the board with empty squares
        for (let i = 0; i < 8; i++) {
            this.board[i] = Array(8).fill(EMPTY);
        }

        // start with two white and two black pieces
        this.board[3][3] = this.board[4][4] = WHITE;
        this.board[3][4] = this.board[4][3] = BLACK;

        this.#updateValidMoves();
    }

    /**
     * the current player makes a move
     * @param {Number} x the x position of the move
     * @param {Number} y the y position of the move
     * @param {-1 | 1} player asserts the correct player is moving
     * @returns {Boolean} ```true``` if the move was valid, ```false``` otherwise
     */
    move(x, y) {
        // find all pieces that would flip
        const toFlip = [];

        for (let direction = 0; direction < 8; direction++) {
            let distance = 0, maybeFlip = [], pos, piece;
            do {
                distance++;
                pos = this.#getPos(x, y, direction, distance);
                piece = this.board[pos[0]]?.[pos[1]];
                
                if (piece === -this.turn) {
                    maybeFlip.push(pos);
                } else if (piece === this.turn) {
                    // flip all those pieces
                    toFlip.push(...maybeFlip);
                }
            } while (piece === -this.turn);
        }

        // flip each piece
        for (const [x, y] of toFlip) {
            this.board[x][y] *= -1;
        }

        if (this.board[x][y] !== EMPTY || toFlip.length === 0) {
            // this should never happen
            throw new Error(`Invalid move: ${x}, ${y}`);
        }

        // pass the turn
        this.turn = -this.turn;

        if (!this.#updateValidMoves()) {
            // if there are no valid moves, the turn passes
            this.turn = -this.turn;
            if (!this.#updateValidMoves()) {
                // if there are still no valid moves, the game ends
                this.isDone = true;
            }
        }
    }

    /**
     * @param {Number} action a number from 0 to 63 saying what square to go in
     * @returns {Boolean} ```true``` if the move was valid, ```false``` otherwise
     */
    doAction(action) {
        return this.move(Math.floor(action / 8), action % 8);
    }

    /** converts the current state of the game into a tensor */
    stateTensor(tensor) {
        let board = this.board;
        if (this.turn === -1) {
            // regardless of color, for the model,
            // "1" is the color making the decision, and "-1" is the other player
            board = board.map(column => column.map(piece => -piece));
        }
        return tensor(board, [8, 8], 'int32');
    }

    /** do a random move */
    randomMove() {
        const actions = [];
        this.validMoves.forEach((isValid, action) => isValid && actions.push(action));
        return actions[Math.floor(Math.random() * actions.length)];
    }

    /** find the best move */
    bestMove(net, state) {
        return net.predict(state) // get Q values for all moves
            .flatten() // flatten to 1d tensor
            .add(this.validMoves.map(a => a === 0 ? -Infinity : 0)) // map invalid moves to -Infinity
            .argMax().dataSync()[0]; // find highest rated move
    }

    #updateValidMoves() {
        this.validMoves = [];
        let anyMoves = false;
        for (let x = 0; x < 8; x++) {
            square: for (let y = 0; y < 8; y++) {
                for (let direction = 0; direction < 8; direction++) {
                    let distance = 0, pos, piece;
                    do {
                        distance++;
                        pos = this.#getPos(x, y, direction, distance);
                        piece = this.board[pos[0]]?.[pos[1]];

                        if (piece === this.turn && distance > 1) {
                            // the move is valid, so stop looking
                            this.validMoves[8 * x + y] = 1;
                            anyMoves = true;
                            continue square;
                        }
                    } while (piece === -this.turn);
                }
                this.validMoves[8 * x + y] = 0;
            }
        }
        return anyMoves;
    }

    static #directions = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];

    #getPos(x, y, direction, distance) {
        const [x2, y2] = Game.#directions[direction];
        return [x + x2 * distance, y + y2 * distance];
    }
}