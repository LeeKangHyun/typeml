import React, { useRef, useEffect } from 'react'

import {
  IMAGE_SIZE,
  mobilenetDemo,
  predict,
} from './Task'

import './index.css'

const ImageClassifier = () => {
  const predEl = useRef<HTMLDivElement>(null)
  const fileEl = useRef<HTMLInputElement>(null)
  const statusEl = useRef<HTMLDivElement>(null);
  const fileEvt = (evt: Event) => {
    let files: FileList | null = (evt.target as HTMLInputElement).files

    // @ts-ignore
    for (let i = 0; i < files.length; i++) {
      let f: File = (files as FileList)[i]
      let reader = new FileReader()
      reader.onload = (e: Event) => {
        let img: HTMLImageElement = document.createElement('img');
        // @ts-ignore
        img.src = e.target.result
        img.width = IMAGE_SIZE
        img.height = IMAGE_SIZE
        img.onload = () => predict(img, predEl.current)
      }

      reader.readAsDataURL(f)
    }
  }
  
  useEffect(() => {
    mobilenetDemo(statusEl).then(() => {
      if (fileEl.current) {
        fileEl.current.addEventListener('change', fileEvt)
      }
    })
    .catch(err => {
      console.log(err)
    })
    
    return () => {
      const { current } = fileEl
      // @ts-ignore
      current.removeEventListener('change', fileEvt)
    }
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
          Upload an image: <input accept="image/*" type="file" ref={fileEl} name="files[]" multiple />
        </div>

        <div id="predictions" ref={predEl}/>
      </section>
    </div>
  )
}

export default ImageClassifier
