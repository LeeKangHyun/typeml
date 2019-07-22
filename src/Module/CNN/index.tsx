import * as React from 'react'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

import { MnistData } from './data'

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

class CNN extends React.Component {
  componentDidMount(): void {
    this.run().catch(err => console.log(err))
  }
  
  run = async () => {
    try {
      const data = new MnistData()
      await data.load()
      await this.showExamples(data)
      
      const model: tf.Sequential = this.getModel()
      tfvis.show.modelSummary({ name: 'Model Architecture' }, model)
      
      await this.train(model, data)
      
      await this.showAccuracy(model, data)
      await this.showConfusion(model, data)
    } catch (e) {
      throw e
    }
  }
  
  showExamples = async (data: any) => {
    const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' })
    
    const examples = data.nextTestBatch(20)
    const numExamples = examples.xs.shape[0]
    
    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tf.tidy(() => {
        return examples.xs.slice([i, 0], [1, examples.xs.shape[1]]).reshape([28, 28, 1])
      })
      
      const canvas: HTMLCanvasElement = document.createElement('canvas') as HTMLCanvasElement
      canvas.width = 28
      canvas.height = 28
      canvas.style.margin = '4px'
      await tf.browser.toPixels(imageTensor, canvas)
      
      surface.drawArea.appendChild(canvas)
      
      imageTensor.dispose()
    }
  }
  
  getModel = (): tf.Sequential => {
    const model = tf.sequential()
    
    const IMAGE_WIDTH = 28
    const IMAGE_HEIGHT = 28
    const IMAGE_CHANNELS = 1
    
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }))
    
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))
    
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }))
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))
    
    model.add(tf.layers.flatten())
    
    const NUM_OUTPUT_CLASSES = 10
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }))
    
    const optimizer = tf.train.adam()
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    })
    
    return model
  }
  
  train = async (model: any, data: any) => {
    const metircs = ['loss', 'val_loss', 'acc', 'val_acc']
    const container = {
      name: 'Model Training',
      styles: {
        height: '1000px'
      }
    }
    
    const fitCallbacks = tfvis.show.fitCallbacks(container, metircs)
    
    const BATCH_SIZE = 512
    const TRAIN_DATA_SIZE = 5500
    const TEST_DATA_SIZE = 1000
    
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ]
    })
    
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE)
      
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ]
    })
    
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    })
  }
  
  doPrediction = (model: tf.Sequential, data: any, testDataSize = 500) => {
    const IMAGE_WIDTH = 28
    const IMAGE_HEIGHT = 28
    const testData = data.nextTestBatch(testDataSize)
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    const labels = testData.labels.argMax(-1)
    const preds = (model.predict(testxs) as tf.Tensor).argMax(-1)
    
    testxs.dispose()
    return [preds, labels]
  }
  
  showAccuracy = async (model: tf.Sequential, data: any) => {
    const [preds, labels] = this.doPrediction(model, data)
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
    const container = {
      name: 'Accuracy',
      tab: 'Evaluation'
    }
    
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames).then(() => {
      labels.dispose()
    })
  }
  
  showConfusion = async (model: tf.Sequential, data: any) => {
    const [preds, labels] = this.doPrediction(model, data)
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds)
    const container = {
      name: 'Confusion Matrix',
      tab: 'Evaluation'
    }
    
    tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames}).then(() => {
      labels.dispose()
    })
    
  }
  
  render() {
    return (
      <div>
      
      </div>
    )
  }
}

export default CNN
