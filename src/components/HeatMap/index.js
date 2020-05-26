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
        heatmapPoints: [],
        marker: false,
        infoWindow: false,
        geocoder: false
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
                    // weight -5 (very positive) to 5 (very negative)
                    weight = Math.min(5,Math.max(-5,-1*weight));
                    if (weight <= 0) {
                        // 100 good articles = 1 bad article in intensity
                        weight = 0.01;
                    }
                    else {
                        weight = weight*10;
                    }
                    this.setState({
                        heatmapPoints: [...this.state.heatmapPoints, 
                            { lat, lng, weight }]
                    });
                }
              this.setState({ articles: data })
              this.setState({marker: new google.maps.Marker});
              this.setState({infoWindow: new google.maps.InfoWindow});
              this.setState({geocoder: new google.maps.Geocoder});
            })
            .catch(console.log)
        )
      };

    onMapClick({ x, y, lat, lng, event }) {
        if (!this.state.heatmapVisible) {
            return
        }
        console.log("You clicked lat=",lat," lng=",lng);
        //window.open('/');

        this.setState({
            heatmapPoints: [...this.state.heatmapPoints, { lat, lng }]
        })
        if (this._googleMap !== undefined) {
            var m = this._googleMap.map_;
            const point = new google.maps.LatLng(lat, lng);
            this.state.marker.setMap(null);
            this.state.infoWindow.close();
            this.state.geocoder.geocode({'location': point}, function (results, status) {
                if (status == 'OK') {
                    var html = results[0].formatted_address;
                    this.state.marker.setPosition(point);
                    this.state.infoWindow.setPosition(point);
                    this.state.marker.setMap(m);
                    this.state.infoWindow.setContent(html);
                    this.state.infoWindow.open(m, this.state.marker);
                    m.panTo(point);
                    console.log(results);
                }
                else { 
                    console.log(status);
                }
            });
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
                <div className="mapTitle">Covid News Explorer</div>
                <button className="toggleButton" onClick={this.toggleHeatMap.bind(this)}>Toggle heatmap</button>
            </div>
        )
    }
}

export default HeatMap;