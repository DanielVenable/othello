import * as tf from '@tensorflow/tfjs-node';
import { SingleBar, Presets } from 'cli-progress';

import { Agent } from './agent.js';
import { Game, BLACK, WHITE } from '../public/game.js';

function train(steps, batchSize, learningRate, syncEveryFrames, replayBufferSize) {
    const game = new Game,
        player1 = new Agent(game, BLACK, replayBufferSize),
        player2 = new Agent(game, WHITE, replayBufferSize),
        optimizer = tf.train.adam(learningRate);

    // show a progress bar
    const replayBar = new SingleBar({
        format: 'Initializing replay buffer: {bar} {value}/{total} | {percentage}% | {eta_formatted} remaining',
        hideCursor: true
    }, Presets.shades_classic);

    replayBar.start(replayBufferSize * 2, 0);

    // fill the replay buffer with random moves without learning
    for (let i = 0; i < replayBufferSize * 2; i++) {
        (game.turn === BLACK ? player1 : player2).doMove();
        replayBar.increment();
    }

    console.log('Replay buffer initialized!');

    // show another progress bar
    const trainingBar = new SingleBar({
        format: 'Training: {bar} {value}/{total} | {percentage}% | {eta_formatted} remaining',
        hideCursor: true
    }, Presets.shades_classic);

    trainingBar.start(steps, 0);

    // do moves and learn from them
    for (let i = 0; i < steps; i++) {
        if (i % syncEveryFrames === 0) {
            player1.updateTarget();
            player2.updateTarget();
        }

        const currentPlayer = game.turn === BLACK ? player1 : player2;

        currentPlayer.trainOnReplayBatch(batchSize, optimizer);

        currentPlayer.doMove();

        if (game.isDone) {
            game.reset();
        }

        trainingBar.increment();
    }

    console.log('Done!');

    return [player1.onlineNN, player2.onlineNN];
}

// adjust these numbers
const [net] = train(1000, 1000, 0.001, 300, 20000);

net.save(`file://${process.cwd()}/public/model`);