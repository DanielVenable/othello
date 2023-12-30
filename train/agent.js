import * as tf from '@tensorflow/tfjs-node';

import { Game } from '../public/game.js';

export class Agent {
    targetNN = createNN();
    onlineNN = createNN();

    /** @type {tf.Tensor2D} */ lastState;
    /** @type {Number} */ lastAction;

    /**
     * @param {Game} game
     * @param {-1 | 1} color black or white
     * @param {Number} epsilon how often to pick a random move
     * @param {Number} replayBufferSize how much game history to store in memory
     */
    constructor(game, color, replayBufferSize, epsilon = 0.1) {
        this.game = game;
        this.color = color;
        this.epsilon = epsilon;
        this.replayMemory = new ReplayMemory(replayBufferSize);
    }
    
    /** pick one move */
    doMove() {
        const state = this.game.stateTensor(tf.tensor2d);

        if (this.lastState) {
            this.replayMemory.append({
                state: this.lastState,
                action: this.lastAction,
                reward: this.reward(state),
                nextState: state,
                isDone: this.game.isDone
            });
        }

        let action;
        if (Math.random() < this.epsilon) {
            // do a random action
            action = this.game.randomMove();
        } else {
            // find the best action
            action = this.game.bestMove(this.onlineNN, tf.stack([state]));
        }

        console.assert(this.game.turn === this.color);
        this.game.doAction(action);

        this.lastState = state;
        this.lastAction = action;
    }

    /**
     * @param {tf.Tensor2D} state 
     * @returns {Number} the reward
     */
    reward(state) {
        return this.game.isDone ? state.sum() : 0;

        // or maybe:
        // return state.sum() - this.lastState.sum();

        // not sure which is better
    }

    /** copy weights from online network to target network */
    updateTarget() {
        this.targetNN.setWeights(this.onlineNN.getWeights());
    }

    /** train the online network on earlier examples */
    trainOnReplayBatch(batchSize, optimizer) {
        const batch = [...this.replayMemory.sample(batchSize)];

        const loss = () => {
            const stateTensor = tf.stack(batch.map(({ state }) => state));
            const actionTensor = tf.tensor1d(batch.map(({ action }) => action), 'int32');

            const online = this.onlineNN.apply(stateTensor, { training: true });
            const oneHot = tf.oneHot(actionTensor, 64);

            // extract the Q values for the actions taken
            const qs = online.mul(oneHot).sum(-1);

            // compute the Q value of the next state.
            // it is R if the next state is terminal
            // R + max Q(next_state) if the next state is not terminal
            const rewardTensor = tf.tensor1d(batch.map(({ reward }) => reward));
            const nextStateTensor = tf.stack(batch.map(({ nextState }) => nextState));

            // target net is used here for stability during training
            const nextMaxQTensor = this.targetNN.predict(nextStateTensor).max(-1);

            // a tensor containing '0' when it is done and '1' otherwise
            const doneMask = tf.tensor1d(batch.map(({ isDone }) => !isDone)).asType('float32');

            const targetQs = rewardTensor.add(nextMaxQTensor.mul(doneMask));

            // define the mean square error between Q value of current state
            // and target Q value
            const mse = tf.losses.meanSquaredError(targetQs, qs);
            return mse;
        }

        tf.tidy(() => {
            // Get a batch of examples from the replay buffer.
            
            // Calculate the gradients of the loss function with respect 
            // to the weights of the online DQN.
            const grads = tf.variableGrads(loss);
            optimizer.applyGradients(grads.grads);
        });
    }
}

function createNN() {
    const net = tf.sequential();

    // input and flatten layers
    net.add(tf.layers.flatten({ inputShape: [8, 8] }));

    // hidden layers
    // may want to adjust these to see what works
    net.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    net.add(tf.layers.dense({ units: 24, activation: 'relu' }));

    // output layer. there are 64 possible actions, so there are 64 units at end
    net.add(tf.layers.dense({ units: 64, activation: 'linear' }));

    net.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    return net;
}

class ReplayMemory {
    index = 0;
    items = [];

    constructor(maxSize) {
        this.maxSize = maxSize;
    }

    /** add an item, removing the first item to enter */
    append(item) {
        if (this.items.length < this.maxSize) {
            this.items.push(item);
        } else {
            this.items[this.index].state.dispose(); // dispose tensor to prevent memory leaks
            this.items[this.index] = item;
            this.index = (this.index + 1) % this.items.length;
        }
    }

    randomItem() {
        return this.items[Math.floor(Math.random() * this.items.length)];
    }

    *sample(num) {
        while (num--) yield this.randomItem();
    }
}