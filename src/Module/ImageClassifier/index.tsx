import * as React from 'react'
import { useRef, useEffect, useCallback, ChangeEvent } from 'react'

import {
  IMAGE_SIZE,
  mobilenetDemo,
  predict,
} from './Task'

import './index.css'

const ImageClassifier = () => {
  const predEl = useRef<HTMLDivElement | null>(null)
  const fileEl = useRef<HTMLInputElement | null>(null)
  const statusEl = useRef<HTMLDivElement | null>(null);
  
  const fileEvt = useCallback((evt: ChangeEvent<HTMLInputElement>) => {
    const { files } = evt.target

    for (let i = 0; i < (files as FileList).length; i++) {
      let data = (files as FileList)[i]
      let reader = new FileReader()
      reader.onload = (ev: ProgressEvent) => {
        let img: HTMLImageElement = document.createElement('img');
        img.src = (ev.target as FileReader).result as string
        img.width = IMAGE_SIZE
        img.height = IMAGE_SIZE
        img.onload = () => predict(img, predEl.current)
      }

      reader.readAsDataURL(data)
    }
  }, [])
  
  useEffect(() => {
    mobilenetDemo(statusEl).catch(err => console.log(err))
  }, [])
  
  return (
    <div className="tfjs-example-container">
      <section>
        <p className='section-head'>상태</p>
        <div id="status" ref={statusEl} />
      </section>

      <section>
        <p className='section-head'>모델</p>

        <div id="file-container">
          Upload an image: <input onChange={fileEvt} accept="image/*" type="file" ref={fileEl} name="files[]" multiple />
        </div>

        <div id="predictions" ref={predEl}/>
      </section>
    </div>
  )
}

export default ImageClassifier
