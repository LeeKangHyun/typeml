import * as React from 'react'
import * as tf from '@tensorflow/tfjs'
import {
  IMAGENET_CLASSES
} from '../file/imagenet_classes'

export const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
export const IMAGE_SIZE = 224
export const TOPK_PREDICTIONS = 10

let mobilenet: tf.LayersModel

let demoStatusElement: HTMLElement

const status = (msg: string) =>  demoStatusElement.innerText = msg

export const mobilenetDemo = async (
  statusEl: React.RefObject<HTMLDivElement>
) => {
  (demoStatusElement as unknown) = await statusEl.current
  
  status('모델 불러오는 중..')
  
  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  
  (mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])) as tf.Tensor).dispose()
  
  status('모델 불러오기 완료')
}

export const predict = async (
  imgElement: HTMLImageElement,
  predictionsElement: Element | any
) => {
  status('결과를 불러오는 중 입니다.')
  
  const startTime1 = performance.now()
  
  let startTime2 = performance.now()
  
  const logits = tf.tidy(() => {
    const img = tf.browser.fromPixels(imgElement).toFloat()
    
    const offset = tf.scalar(127.5)
    
    const normalized = img.sub(offset).div(offset)
    
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3])
    
    startTime2 = performance.now()
    
    return mobilenet.predict(batched)
  })
  
  const classes = await getTopKClasses(logits as tf.Tensor, TOPK_PREDICTIONS)
  const totalTime1 = performance.now() - startTime1
  const totalTime2 = performance.now() - startTime2
  
  status(`완료시간 ${Math.floor(totalTime1)} ms\n(전처리를 제외한 시간: ${Math.floor(totalTime2)} ms)`)
  
  showResults(imgElement, predictionsElement, classes)
}

type topKClass = {
  value: number
  index: number
}

export const getTopKClasses = async (
  logits: tf.Tensor,
  topK: number
) => {
  const values = await logits.data()
  
  const valuesAndIndices: topKClass[] = []
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i })
  }
  
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value
  })
  
  const topkValues = new Float32Array(topK)
  const topkIndices = new Int32Array(topK)
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value
    topkIndices[i] = valuesAndIndices[i].index
  }
  
  const topClassesAndProbs = []
  
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  
  return topClassesAndProbs
}

type Classes = {
  className: string
  probability: number
}

export const showResults = (
  imgElement: HTMLImageElement,
  predictionsElement: HTMLDivElement,
  classes: Classes[]
) => {
  const predictionContainer = document.createElement('div')
  predictionContainer.className = 'pred-container'
  
  const imgContainer = document.createElement('div')
  imgContainer.appendChild(imgElement)
  predictionContainer.appendChild(imgContainer)
  
  const probsContainer = document.createElement('div')
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div')
    row.className = 'row'
    
    const classElement = document.createElement('div')
    classElement.className = 'cell'
    classElement.innerText = classes[i].className
    row.appendChild(classElement)
    
    const probsElement = document.createElement('div')
    probsElement.className = 'cell'
    probsElement.innerText = (classes[i].probability * 100).toFixed(3) + '%'
    row.appendChild(probsElement)
    
    probsContainer.appendChild(row)
  }
  predictionContainer.appendChild(probsContainer);
  
  predictionsElement.insertBefore(
    predictionContainer, predictionsElement.firstChild
  )
}
