"use strict";
exports.__esModule = true;
var tf = require("@tensorflow/tfjs");
var model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
var xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
var ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
model.fit(xs, ys, { epochs: 10 }).then(function () {
    var MD = model.predict(tf.tensor2d([5], [1, 1]));
    console.log(MD);
});
