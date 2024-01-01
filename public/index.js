import { Game, BLACK, WHITE } from './game.js';

const load = new Promise(resolve => document.querySelector('#board').addEventListener('load', resolve));

async function play(modelName) {
    document.querySelector('#buttons').hidden = true;
    document.querySelector('#loading').hidden = false;

    const playerTurn = BLACK;
    const botTurn = WHITE;
    const model = await tf.loadLayersModel(`models/${modelName}/model.json`);
    const game = new Game;

    await load;
    const board = document.querySelector('#board').contentDocument;

    document.querySelector('#loading').hidden = true;
    document.querySelector('#disable').hidden = true;

    board.addEventListener('click', ({ target }) => {
        if (game.turn === playerTurn) {
            const x = +target.getAttribute('x'),
                y = +target.getAttribute('y');
            if (game.validMoves[x * 8 + y]) {
                goOn(x, y);
            }
        }
    });

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
            setTimeout(() => {
                let action;
                tf.tidy(() => {
                    action = game.bestMove(model, tf.stack([game.stateTensor(tf.tensor2d)]));
                });

                setTimeout(async () => {
                    const square = await action;
                    goOn(Math.floor(square / 8), square % 8);
                }, 200);
            }, 0);
        } else if (game.isDone) {
            document.querySelector('#disable').hidden = false;
            document.querySelector('#done').hidden = false;

            const [blackScore, whiteScore] = game.scores();
            document.querySelector('#blackScore').textContent = blackScore;
            document.querySelector('#whiteScore').textContent = whiteScore;
        }
    }
}

for (const btn of document.querySelectorAll('#buttons button')) {
    btn.addEventListener('click', () => play(btn.dataset.modelName));
}