import * as tf from '@tensorflow/tfjs-node';
import { SingleBar, Presets } from 'cli-progress';

import { Agent } from './agent.js';
import { Game, BLACK, WHITE } from '../public/game.js';

async function train(steps, batchSize, learningRate, syncEveryFrames, replayBufferSize, ...config) {
    const game = new Game,
        player1 = new Agent(game, BLACK, replayBufferSize, ...config),
        player2 = new Agent(game, WHITE, replayBufferSize, ...config),
        optimizer = tf.train.adam(learningRate);

    const barFormat = '{bar} {value}/{total} | {percentage}% | {eta_formatted} remaining';

    // show a progress bar
    const replayBar = new SingleBar({
        format: 'Initializing replay buffer: ' + barFormat,
        hideCursor: true
    }, Presets.shades_classic);

    replayBar.start(replayBufferSize * 2, 0);

    // fill the replay buffer with random moves without learning
    for (let i = 0; i < replayBufferSize * 2; i++) {
        await (game.turn === BLACK ? player1 : player2).doMove();
        if (game.isDone) {
            game.reset();
        }
        replayBar.increment();
    }

    replayBar.stop();

    console.log('Replay buffer initialized.');

    // show another progress bar
    const trainingBar = new SingleBar({
        format: 'Training: ' + barFormat,
        hideCursor: true
    }, Presets.shades_classic);

    trainingBar.start(steps, 0);

    // do moves and learn from them
    for (let i = 0; i < steps; i++) {
        if (i % syncEveryFrames === 0) {
            player1.updateTarget();
            player2.updateTarget();
        }

        // make sure the model is not lost if the program crashes
        if (i % 200000 === 199999) save();

        const currentPlayer = game.turn === BLACK ? player1 : player2;

        currentPlayer.trainOnReplayBatch(batchSize, optimizer);

        await currentPlayer.doMove();

        if (game.isDone) {
            game.reset();
        }

        trainingBar.increment();
    }

    trainingBar.stop();

    console.log('Done!');

    return [player1.onlineNN, player2.onlineNN];
}

function save() {
    return net.save(`file://${process.cwd()}/public/models/${process.env.MODEL_NAME ?? 'model'}`);
}

const [net] = await train(
    process.env.STEPS ?? 100000,
    process.env.BATCH_SIZE ?? 1000,
    process.env.LEARNING_RATE ?? 0.01,
    process.env.SYNC_FREQ ?? 300,
    process.env.REPLAY_BUFFER_SIZE ?? 20000,
    !!+(process.env.REWARD_EVERY_STEP ?? 1),
    process.env.EPSILON ?? 0.1,
    +(process.env.HIDDEN_LAYERS ?? 2),
    +(process.env.UNITS ?? 24));

await save();

console.log('Model saved.');

process.exit();