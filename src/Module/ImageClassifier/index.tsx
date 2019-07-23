import React, { createRef } from 'react'

import {
  IMAGE_SIZE,
  mobilenetDemo,
  predict
} from './Task'

import './index.css'

class ImageClassifier extends React.Component {
  private fileEl = createRef<HTMLInputElement>()
  private predEl = createRef<HTMLDivElement>()
  
  componentDidMount(): void {
    mobilenetDemo().then(() => {
      this.makeImg()
    })
    .catch(err => console.log(err))
  }
  
  makeImg = () => {
    if (this.fileEl.current && this.predEl.current) {
      this.fileEl.current.addEventListener('change', (evt: Event) => {
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
            img.onload = () => predict(img, this.predEl.current)
          }
      
          reader.readAsDataURL(f)
        }
      })
    }
  }
  
  render() {
    return (
      <div className="tfjs-example-container">
        <section>
          <p className='section-head'>Status</p>
          <div id="status" />
        </section>

        <section>
          <p className='section-head'>Model Output</p>

          <div id="file-container">
            Upload an image: <input accept="image/*" type="file" ref={this.fileEl} name="files[]" multiple />
          </div>

          <div id="predictions" ref={this.predEl}/>
        </section>
      </div>
    )
  }
}

export default ImageClassifier
