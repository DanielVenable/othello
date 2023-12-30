import { Game, BLACK, WHITE } from './game.js';

const loadModel = tf.loadLayersModel('model/model.json');
document.querySelector('object').addEventListener('load', async function () {
    const playerTurn = BLACK;
    const botTurn = WHITE;
    const model = await loadModel;
    const game = new Game;
    const board = this.contentDocument;

    board.addEventListener('click', ({ target }) => {
        if (game.turn === playerTurn) {
            const x = +target.getAttribute('x'),
                y = +target.getAttribute('y');
            if (game.validMoves[x * 8 + y]) {
                goOn(x, y);
            }
        }
    });

    function botGo() {
        const action = game.bestMove(model, tf.stack([game.stateTensor(tf.tensor2d)]));
        goOn(Math.floor(action / 8), action % 8);
    }

    function goOn(x, y) {
        const color = game.turn;
        const toFlip = game.move(x, y);

        board.querySelector(`use[x="${x}"][y="${y}"]`)
            .setAttribute('href', color === BLACK ? '#black' : '#white');

        for (const [x, y] of toFlip) {
            board.querySelector(`use[x="${x}"][y="${y}"]`)
                .setAttribute('href', color === BLACK ? '#black' : '#white');
        }
        
        if (game.turn === botTurn) {
            setTimeout(botGo, 250);
        }
    }
});