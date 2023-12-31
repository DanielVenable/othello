/* This plays different models against each other to see who wins. Use to figure out how to best train models */

import * as tf from '@tensorflow/tfjs-node';
import { Game, BLACK } from './public/game.js';

const names = process.argv.slice(2);

if (names.length < 2) {
    console.error('Please select at least 2 models to compare.');
    console.error('You can specify models like this: "npm run compare easy medium"');
    process.exit(1);
}

// load all models
const models = await Promise.all(names.map(name =>
    tf.loadLayersModel(`file://${process.cwd()}/public/models/${name}/model.json`)));

const game = new Game,
    scores = Array(models.length).fill(0);

for (let i = 0; i < models.length; i++) {
    for (let j = 0; j < models.length; j++) {
        if (i === j) continue; // no need for a model to play itself

        const net1 = models[i], net2 = models[j];

        while (!game.isDone) {
            const state = game.stateTensor(tf.tensor2d);
            game.doAction(game.bestMove(game.turn === BLACK ? net1 : net2, tf.stack([state])));
            state.dispose();
        }

        const [score1, score2] = game.scores();
        scores[i] += score1;
        scores[j] += score2;

        game.reset();
    }
}

console.log('Final average scores:');

const gamesPlayed = models.length ** 2 - models.length;

// sort by score descending
for (const key of [...scores.keys()].sort((a, b) => scores[a] - scores[b])) {
    console.log('%s: %f', names[key], (scores[key] / gamesPlayed).toFixed(2));
}