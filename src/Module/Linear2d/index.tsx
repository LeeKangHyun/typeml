import * as React from 'react'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

type Car = {
  Acceleration: number
  Cylinders: number
  Displacement: number
  Horsepower: number
  Miles_per_Gallon: number
  Name: string
  Origin: string
  Weight_in_lbs: number
  Year: string
}

type DataType = {
  horsepower: number
  mpg: number
}

class App extends React.Component {
  componentDidMount(): void {
    this.run().catch(err => console.log(err))
  }
  
  getData = async () => {
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const carsData = await carsDataReq.json()
    return carsData.map((car: Car): DataType => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car: DataType) => (car.mpg != null && car.horsepower != null))
  }
  
  run = async () => {
    try {
      const data = await this.getData()
      const values = data.map((d: DataType) => ({
        x: d.horsepower,
        y: d.mpg
      }))
      const tensorData = await this.convertToTensor(data)
      const { inputs, labels } = tensorData
      
      const model = await this.createModel()
      await this.trainModel(model, inputs, labels)
      await this.testModel(model, data, tensorData)
      
      
      
      tfvis.show.modelSummary({ name: 'Model Summary' }, model)
      
      tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
          xLabel: 'Horsepower',
          yLabel: 'MPG',
          height: 300
        }
      )
      
      console.log('Done Training')
      
    } catch (e) {
      throw new Error(e)
    }
  }
  
  createModel = () => {
    const model = tf.sequential()
    
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
    model.add(tf.layers.dense({ units: 1, useBias: true }))
    
    return model
  }
  
  convertToTensor = (data: DataType[]) => {
    return tf.tidy(() => {
      tf.util.shuffle(data)
      
      const inputs = data.map((d: DataType) => d.horsepower)
      const labels = data.map((d: DataType) => d.horsepower)
      
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
      const labelTensor = tf.tensor2d(labels, [labels.length, 1])
      
      const inputMax = inputTensor.max()
      const inputMin = inputTensor.min()
      const labelMax = labelTensor.max()
      const labelMin = labelTensor.min()
      
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))
      
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin
      }
    })
  }
  
  trainModel = async (model: any, inputs: any, labels: any) => {
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse']
    })
    
    const batchSize = 32
    const epochs = 50
    
    return await model.fit(inputs, labels, {
      batchSize: batchSize,
      epochs: epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        {
          name: 'Training Performance'
        },
        ['loss', 'mse'],
        {
          height: 200,
          callbacks: ['onEpochEnd']
        }
      ) as any
    })
  }
  
  testModel = (model: any, inputData: DataType[], normalizationData: any) => {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData
    
    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100)
      const preds = model.predict(xs.reshape([100, 1]))
      
      const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin)
      const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)
      
      return [unNormXs.dataSync(), unNormPreds.dataSync()]
    })
    
    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] }
    })
    
    const originalPoints = inputData.map((d: DataType) => ({
      x: d.horsepower,
      y: d.mpg
    }))
    
    tfvis.render.scatterplot(
      { name: 'Model Predictions vs Original Data' },
      {
        values: [ originalPoints, predictedPoints ],
        series: [ 'original', 'predicted' ]
      },
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    )
  }
  
  render() {
    return (
      <div />
    )
  }
}

export default App;
