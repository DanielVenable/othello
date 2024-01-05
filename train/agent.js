import * as tf from '@tensorflow/tfjs-node';

import { Game } from '../public/game.js';

export class Agent {
    /** @type {tf.Tensor2D} */ lastState;
    /** @type {Number} */ lastAction;

    /**
     * @param {Game} game
     * @param {-1 | 1} color black or white
     * @param {ReplayMemory} replayMemory game history for learning
     * @param {Boolean} rewardEveryStep does it give rewards after every step or only at the end
     * @param {Number} epsilon how often to pick a random move
     * @param {Number} onlineNN the online net
     * @param {Number} targetNN the target net
     */
    constructor(...args) {
        [this.game, this.color, this.replayMemory, this.rewardEveryStep, this.epsilon,
            this.onlineNN, this.targetNN] = args;
    }
    
    /** pick one move */
    async doMove() {
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
            tf.tidy(() => {
                action = this.game.bestMove(this.onlineNN, tf.stack([state]));
            });
            action = await action;
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
        if (this.rewardEveryStep) {
            return tf.tidy(() => state.sum().sub(this.lastState.sum()));
        } else {
            return this.game.isDone ? state.sum() : tf.scalar(0);
        }
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
            const rewardTensor = tf.stack(batch.map(({ reward }) => reward));
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
            // Calculate the gradients of the loss function with respect 
            // to the weights of the online DQN.
            optimizer.applyGradients(tf.variableGrads(loss).grads);
        });
    }
}