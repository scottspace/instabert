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
        geocoder: false,
        index: 0, 
        timer: false
      };

    myTimer() {
        if (this.state.timer) {
            clearTimeout(this.state.timer);
        }
        var idx = this.state.index;
        var m_idx = this.state.articles.length;
        idx += 1;
        if (idx > m_idx) {
            idx = 0;
        }
        this.setState({index: idx});
        console.log("We got you - index at "+idx);
        console.log(this.state.articles[idx]);
        const new_timer = setTimeout(() => {
            this.myTimer();
            }, 5000);
        this.setState({timer: new_timer});
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
              this.setState({index: 0});
              this.myTimer();
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
            var marker = this.state.marker;
            var infoWindow = this.state.infoWindow;
            const point = new google.maps.LatLng(lat, lng);
            marker.setMap(null);
            infoWindow.close();
            this.state.geocoder.geocode({'location': point}, function (results, status) {
                if (status == 'OK') {
                    var raw = results[0].formatted_address;
                    var parts = raw.split(',')
                    var city = parts[1].trim();
                    var city2 = city.replace(' ','+');
                    var html = "See news about <a href=\"/?c="+city2+"\">"+city+"</a>";
                    marker.setPosition(point);
                    infoWindow.setPosition(point);
                    marker.setMap(m);
                    infoWindow.setContent(html);
                    infoWindow.open(m, marker);
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