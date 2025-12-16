import React from "react";
import "../App.css";
import LineChart from "./LineChart";
import Label from "./AxisLabel";
import ChartTitle from "./ChartTitle";

const styles = {
  chartComponentsContainer: {
    display: 'grid', gridTemplateColumns: 'max-content 480px', alignItems: 'center'
  },
  chartWrapper: { maxWidth: 700, alignSelf: 'flex-start' }
}

function YieldCurve({data}) {
  
  if (data != null) {
    return (
      <div style={styles.chartComponentsContainer}>
        <div/>
        <ChartTitle text="Yield Curve"/>
        <Label text="Yield" rotate/>
        <div style={styles.chartWrapper}>
          <LineChart
          width={800 }
            height={500}
            data={data}
            horizontalGuides={5}
            precision={2}
            verticalGuides={1}
          />
        </div>
        <div/>
        <Label text="Years"/>
      </div>
    );
    }
    else{
      return(<h2>wait</h2>)
    }
}

export default YieldCurve;