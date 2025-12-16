/*
 * @Date: 2021-09-29 13:05:21 
 */
import React from "react";
import "../App.css";
import Plot from 'react-plotly.js';

function YTWCurve({ytw_curve_data}) {
  var tick = (Math.max.apply(Math,ytw_curve_data.dd)-Math.min.apply(Math,ytw_curve_data.dd)) 

    return (
      <Plot
        data={[
          {type: 'line', 
          name: 'Dealer Dealer',
            x: ytw_curve_data.x,
            y: ytw_curve_data.dd,
            line: {
              color: '#43e4ff',
              width: 3}},
          {type: 'line', 
          name: 'Dealer Sells',
            x: ytw_curve_data.x,
            y: ytw_curve_data.ds,
            line: {
              color: 'rgb(55,155,255)',
              width: 3}},       
          {type: 'line', 
          name: 'Dealer Purchase',
            x: ytw_curve_data.x,
            y: ytw_curve_data.dp,
            line: {
              color: 'rgb(235,171,184)',
              width: 3}}
        ]}
        layout={ {showlegend: true,   legend: {},displaylogo: false,
        width: 500,
        height: 400,
        paper_bgcolor:'rgb(48,48,48)',
        plot_bgcolor: 'rgb(48,48,48)',
        font: {
          family: 'Helvetica,Arial',
          size: 18,
          color: '#59e4fd'
        },
        line: {
          color: 'rgb(55, 128, 191)',
          width: 3
        },
        useResizeHandler: true,
        margin: {
          l: 0,
          r: 0,
          t: 0,
          b: 0,
          pad: 0,
          autoexpand: true
        },
        margin: {
          t: 0
        },
        legend: {
          xanchor: 'right',
          yanchor: 'bottom'
        },
        autosize: true, // set autosize to rescale
        xaxis: {
          rangemode:'normal',
          autorange: true,
          title: 'Trade Quantity ($)'
        },
        yaxis: {
          rangemode:'normal',
          autorange: true,
          title: 'Yield',
          dtick: tick,
          hoverformat: ',.3f'
        } } }
      />
    );
}

export default YTWCurve;