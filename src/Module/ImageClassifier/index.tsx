import * as React from 'react'
import * as mobilenet from '@tensorflow-models/mobilenet'

class ImageClassifier<P> extends React.Component<P> {
  net: mobilenet.MobileNet | null
  el: HTMLImageElement | null
  
  constructor(props: P) {
    super(props)
    this.net = null
  }
  
  componentDidMount(): void {
    this.app().catch(err => console.log(err))
  }
  
  app = async () => {
    try {
      console.log('Loading...')
  
      this.net = await mobilenet.load({
        version: 1,
        alpha: 1.0
      })
      console.log('Success...')
  
      const result = await this.net.classify(this.el as HTMLImageElement)
      console.log(result)
    } catch(err) {
      throw err
    }
  }
  
  render() {
    return (
      <div>
        <img ref={el => this.el = el} alt="" id="img" src="./image/JlUvsxa.jpg" />
      </div>
    )
  }
}

export default ImageClassifier
