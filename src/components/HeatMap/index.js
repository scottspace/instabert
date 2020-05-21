/* global google */
import React, { Component } from 'react'
import GoogleMapReact from 'google-map-react'

import './HeatMap.css'

class HeatMap extends Component {
    static defaultProps = {
        center: {
            lat: 39.33,
            lng: -120.18
        },
        zoom: 11
    }

    constructor(props) {
        super(props)
        this.state = {
            heatmapVisible: true,
            heatmapPoints: [
                { lat: 39.33, lng: -120.18, weight: 1 },
                { lat: 32.3, lng: -115, weight: 5 }
            ]
        }
    }

    onMapClick({ x, y, lat, lng, event }) {
        if (!this.state.heatmapVisible) {
            return
        }

        this.setState({
            heatmapPoints: [...this.state.heatmapPoints, { lat, lng }]
        })
        if (this._googleMap !== undefined) {
            const point = new google.maps.LatLng(lat, lng)
            this._googleMap.heatmap.data.push(point)
        }
    }

    toggleHeatMap() {
        this.setState({
            heatmapVisible: !this.state.heatmapVisible
        }, () => {
            if (this._googleMap !== undefined) {
                this._googleMap.heatmap.setMap(this.state.heatmapVisible ? this._googleMap.map_ : null)
            }
        })

    }

    render() {

        // sorry, but this key only works on my domain
        const apiKey = { key: 'AIzaSyAoAJ_mPrHFvnhimkLOBrFMSQqEwpkJRCk' }
        const heatMapData = {
            positions: this.state.heatmapPoints,
            options: {
                radius: 20,
                opacity: 0.6
            }
        }

        console.log(this.state)

        return (
            <div style={{ height: '100vh', width: '100%' }}>
                <GoogleMapReact
                    ref={(el) => this._googleMap = el}
                    bootstrapURLKeys={apiKey}
                    defaultCenter={this.props.center}
                    defaultZoom={this.props.zoom}
                    heatmapLibrary={true}
                    heatmap={heatMapData}
                    onClick={this.onMapClick.bind(this)}
                >
                </GoogleMapReact>
                <button className="toggleButton" onClick={this.toggleHeatMap.bind(this)}>Toggle heatmap</button>
            </div>
        )
    }
}

export default HeatMap;