/* global google */
import React, { Component } from 'react'
import GoogleMapReact from 'google-map-react'
import LoadingIndicator from '../LoadingIndicator'
import { trackPromise } from 'react-promise-tracker'

import './HeatMap.css'

function createMapOptions(maps) {
    // next props are exposed at maps
    // "Animation", "ControlPosition", "MapTypeControlStyle", "MapTypeId",
    // "NavigationControlStyle", "ScaleControlStyle", "StrokePosition", "SymbolPath", "ZoomControlStyle",
    // "DirectionsStatus", "DirectionsTravelMode", "DirectionsUnitSystem", "DistanceMatrixStatus",
    // "DistanceMatrixElementStatus", "ElevationStatus", "GeocoderLocationType", "GeocoderStatus", "KmlLayerStatus",
    // "MaxZoomStatus", "StreetViewStatus", "TransitMode", "TransitRoutePreference", "TravelMode", "UnitSystem"
    return {
      zoomControlOptions: {
        position: maps.ControlPosition.RIGHT_CENTER,
        style: maps.ZoomControlStyle.SMALL
      },
      mapTypeControlOptions: {
        position: maps.ControlPosition.TOP_RIGHT
      },
      mapTypeControl: true
    };
  }

class HeatMap extends Component {

    static defaultProps = {
        center: {
            lat: 38.08,
            lng: -119.75
        },
        zoom: 5
    }

    state = {
        articles: [],
        query: '',
        heatmapVisible: true,
        heatmapPoints: []
      };
    
    componentDidMount() {
        var q = this.props.location.search;
        var url = 'https://us-central1-octo-news.cloudfunctions.net/articles'
        var m = '?m=all'
        if (q.length > 0) {
            m = q;
        }  
        url += m
        this.setState({'query': q});
        trackPromise(
          fetch(url)
            .then(res => res.json())
            .then((data) => {
                for (var i=0; i < data.length; i++) {
                    var article = data[i];
                    var weight = article.z;
                    var lat = article.lat;
                    var lng = article.lng;
                    weight = 3*(36-(3+Math.min(3,Math.max(-3,weight))**2));
                    this.setState({
                        heatmapPoints: [...this.state.heatmapPoints, 
                            { lat, lng, weight }]
                    });
                }
              this.setState({ articles: data })
            })
            .catch(console.log)
        )
      };

    onMapClick({ x, y, lat, lng, event }) {
        if (!this.state.heatmapVisible) {
            return
        }
        console.log("You clicked lat=",lat," lng=",lng);
        window.open('/');

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
                    options={createMapOptions}
                    onClick={this.onMapClick.bind(this)}
                >
                </GoogleMapReact>
                <LoadingIndicator />
                <button className="toggleButton" onClick={this.toggleHeatMap.bind(this)}>Toggle heatmap</button>
            </div>
        )
    }
}

export default HeatMap;