import { dispose } from "@tensorflow/tfjs-node";

export class ReplayMemory {
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
            // dispose tensors to prevent memory leaks
            dispose([this.items[this.index].state, this.items[this.index].reward]);

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