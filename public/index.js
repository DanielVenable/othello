import { Game, BLACK, WHITE } from './game.js';

const load = new Promise(resolve => document.querySelector('#board').addEventListener('load', resolve));

let started = false;
async function play(modelName) {
    if (started) return;
    started = true;

    document.querySelector('#buttons').hidden = true;
    document.querySelector('#loading').hidden = false;

    const playerTurn = BLACK;
    const botTurn = WHITE;
    const model = await tf.loadLayersModel(`models/${modelName}/model.json`);
    const game = new Game;

    // warm up the model
    tf.tidy(() => {
        game.bestMove(model, tf.stack([game.stateTensor(tf.tensor2d)]));
    });

    // make sure the board is loaded
    await load;

    // done loading!
    document.querySelector('#loading').hidden = true;
    document.querySelector('#disable').hidden = true;

    const board = document.querySelector('#board').contentDocument;

    updateValid();

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

        flip(toFlip, color === BLACK);

        if (color === playerTurn) {
            removeValid();
        }

        if (game.turn === botTurn) {
            setTimeout(async () => {
                let action;
                tf.tidy(() => {
                    action = game.bestMove(model, tf.stack([game.stateTensor(tf.tensor2d)]));
                });
                const square = await action;
                goOn(Math.floor(square / 8), square % 8);
            }, 200);
        } else if (game.turn === playerTurn) {
            updateValid();
        } else {
            document.querySelector('#disable').hidden = false;
            document.querySelector('#done').hidden = false;

            const [blackScore, whiteScore] = game.scores();
            document.querySelector('#blackScore').textContent = blackScore;
            document.querySelector('#whiteScore').textContent = whiteScore;
        }
    }

    function flip(pieces, isToBlack) {
        const duration = 200;
        const ellipse = board.querySelector('#flip ellipse');
        ellipse.setAttribute('fill', isToBlack ? 'white' : 'black');

        let flipped = false;
        let beginTime = performance.now();

        changeHref('#flip');
        requestAnimationFrame(frame);

        function changeHref(value) {
            for (const [x, y] of pieces) {
                board.querySelector(`use[x="${x}"][y="${y}"]`).setAttribute('href', value);
            }
        }

        function frame() {
            const delta = performance.now() - beginTime;
            ellipse.setAttribute('ry', 0.45 * Math.cos(delta * Math.PI / duration));
            if (!flipped && delta >= duration / 2) {
                ellipse.setAttribute('fill', isToBlack ? 'black' : 'white');
                flipped = true;
            }

            if (delta >= duration) {
                changeHref(isToBlack ? '#black' : '#white');
            } else {
                requestAnimationFrame(frame);
            }
        }
    }

    function updateValid() {
        game.validMoves.forEach((isValid, action) => {
            if (isValid) {
                board.querySelector(`use[x="${Math.floor(action / 8)}"][y="${action % 8}"]`)
                    .setAttribute('href', '#valid');
            }
        });
    }

    function removeValid() {
        for (const square of board.querySelectorAll('use[href="#valid"]')) {
            square.setAttribute('href', '#square');
        }
    }
}

for (const btn of document.querySelectorAll('#buttons button')) {
    btn.addEventListener('click', () => play(btn.dataset.modelName));
}