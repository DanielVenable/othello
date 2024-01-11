import * as tf from '@tensorflow/tfjs-node';
import { SingleBar, Presets } from 'cli-progress';

import { Agent } from './agent.js';
import { ReplayMemory } from './replayMemory.js';
import { Game, BLACK, WHITE } from '../public/game.js';

async function train({
        STEPS = 100000, BATCH_SIZE = 1000, LEARNING_RATE = 0.01, SYNC_FREQ = 300,
        REPLAY_BUFFER_SIZE = 20000, PLAY_ITSELF = 1, REWARD_EVERY_STEP = 1,
        EPSILON = 0.1, HIDDEN_LAYERS = 2, UNITS = 24, FROM_MODEL }) {

    async function getNet() {
        return FROM_MODEL ?
            await tf.loadLayersModel(
                `file://${process.cwd()}/public/models/${FROM_MODEL}/model.json`) :
            createNN(+HIDDEN_LAYERS, +UNITS);
    }

    const replayMemory = new ReplayMemory(REPLAY_BUFFER_SIZE);

    const onlineNet1 = await getNet(),
        targetNet1 = await getNet();

    const game = new Game;

    const player1 = new Agent(game, BLACK, replayMemory,
        !!+REWARD_EVERY_STEP, EPSILON, onlineNet1, targetNet1);

    const player2 = new Agent(game, WHITE, replayMemory, !!+REWARD_EVERY_STEP, EPSILON,
        !!+PLAY_ITSELF ? onlineNet1 : await getNet(),
        !!+PLAY_ITSELF ? targetNet1 : await getNet());

    const optimizer = tf.train.adam(+LEARNING_RATE);

    const barFormat = '{bar} {value}/{total} | {percentage}% | {eta_formatted} remaining';

    // show a progress bar
    const replayBar = new SingleBar({
        format: 'Initializing replay buffer: ' + barFormat,
        hideCursor: true
    }, Presets.shades_classic);

    replayBar.start(REPLAY_BUFFER_SIZE, 0);

    // fill the replay buffer with random moves without learning
    for (let i = 0; i < REPLAY_BUFFER_SIZE; i++) {
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

    trainingBar.start(STEPS, 0);

    // do moves and learn from them
    for (let i = 0; i < STEPS; i++) {
        if (i % SYNC_FREQ === 0) {
            player1.updateTarget();
            player2.updateTarget();
        }

        // make sure the model is not lost if the program crashes
        if (i % 150000 === 149999) await save(player1.onlineNN);

        const currentPlayer = game.turn === BLACK ? player1 : player2;

        currentPlayer.trainOnReplayBatch(BATCH_SIZE, optimizer);

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

function save(net) {
    return net.save(`file://${process.cwd()}/public/models/${process.env.MODEL_NAME ?? 'model'}`);
}

function createNN(hiddenLayers, units) {
    const net = tf.sequential();

    // input and flatten layers
    net.add(tf.layers.flatten({ inputShape: [8, 8] }));

    // hidden layers
    for (let i = 0; i < hiddenLayers; i++) {
        net.add(tf.layers.dense({ units, activation: 'relu' }));
    }

    // output layer. there are 64 possible actions, so there are 64 units at end
    net.add(tf.layers.dense({ units: 64, activation: 'linear' }));

    net.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    return net;
}

const [net] = await train(process.env);

await save(net);

console.log('Model saved.');

process.exit();