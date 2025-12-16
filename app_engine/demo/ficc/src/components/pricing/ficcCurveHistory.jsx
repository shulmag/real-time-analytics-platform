/*
 * @Date: 2023-01-23 
 */
import React from 'react'
import Table from 'react-bootstrap/Table'

import { tradeTypeDict } from './relatedVarDict'

import { useEffect, useState } from 'react'


function FiccCurveHistory({data, cusipForDisplay}) {
    const curve_history_data = data
    // idx_to_feature = {0: 'yield_spread', 
    //                   1: 'ficc_ycl', 
    //                   3: 'yield_to_worst', 
    //                   4: 'dollar_price', 
    //                   6: 'size', 
    //                   7: 'calc_date'
    //                   12: 'trade_datetime', 
    //                   15: 'trade_type'}

    const NO_YIELDS_REPORTED = 'No yields reported'
    const NO_DOLLAR_PRICES_REPORTED = 'No dollar prices reported'

    const [hiddenRows, setHiddenRows] = useState([])

    const [hover, setHover] = useState(false)

    const handleMouseEnter = (e) => {
        e.currentTarget.style.backgroundColor = '#d9edf7'
        setHover(true)
    }
    
    const handleMouseLeave = (e) => {
        e.currentTarget.style.backgroundColor = 'white'
        setHover(false)
    }
    
    const hoverStyle = {
        cursor: hover ? 'pointer' : 'default'
    }

    // This is for debugging since this useEffect does nothing besides logging
    useEffect(() => {
        console.log('hiddenRows changed:', hiddenRows)
      }, [hiddenRows])

    function handleClick(e) {
        const rowId = e.currentTarget.dataset.id
        setHiddenRows(prevHiddenRows => {
            const newHiddenRows = [...prevHiddenRows]
            const index = newHiddenRows.indexOf(rowId)
            if (index === -1) { newHiddenRows.push(rowId) }
            else { newHiddenRows.splice(index, 1) }
            return newHiddenRows
        })
    }

    function dispSingleTradeDetails(grouped, date) {        
        if (true) {    // need to check that curve_history_data has data in it at some point
            return (
                grouped[date].map((c) => (
                    <tr>
                    <td style={{whiteSpace:'nowrap'}}>{c.trade_datetime}</td>
                    {/* <td>{Math.round(c.yield_spread, 0) }</td> */}
                    {/* <td>{Math.round(c.ficc_ycl, 0)}</td> */}
                    <td>{c.yield_to_worst === null ? 'No yield reported' : c.yield_to_worst}</td>
                    <td>{c.dollar_price === null ? 'No dollar price reported': c.dollar_price}</td>
                    <td>{c.calc_date === null? 'No yield reported': c.calc_date}</td>
                    <td>{c.size / 1000}</td>
                    <td style={{whiteSpace:'nowrap'}}>{tradeTypeDict[c.trade_type]}</td>            
                    </tr>
                ))
            )
        }
    }

    function dispTradesStats() {
        // this gives an object with dates as keys
        // https://stackoverflow.com/questions/55272682/reduce-function-with-bracket-key
        function groupBy(objectArray, property) {
            if (objectArray !== undefined) {
                return objectArray.reduce(function(acc, obj) {
                    var key = obj[property]
                    if (!acc[key]) {
                        acc[key] = []
                    }
                    acc[key].push(obj)
                    return acc
                
                }, {})
            }
        }

        if (curve_history_data !== undefined) {
            for (var i = 0; i < curve_history_data.length; i++) {  
                curve_history_data[i]['date'] = curve_history_data[i].trade_datetime.split(' ')[0]
            }
        }

        function total(dailyTradeArr) {
            return dailyTradeArr.reduce((accumulator, trade) => accumulator + trade.size, 0)
        }

        function count(dailyTradeArr) {
            return dailyTradeArr.reduce((accumulator, trade) => accumulator + 1, 0)
        }

        // function high_spread(dailyTradeArr) {
        //     return dailyTradeArr.reduce((accumulator, trade) => Math.max(accumulator, trade.yield_spread), Number.MIN_VALUE)
        // }

        // function low_spread(dailyTradeArr) {
        //     return dailyTradeArr.reduce((accumulator, trade) => Math.min(accumulator, trade.yield_spread), Number.MAX_VALUE)
        // }

        function accumulate_high_yield_ignore_null(accumulator, trade) {
            if (trade.yield_to_worst === null) {
                return accumulator
            } else {
                return Math.max(accumulator, trade.yield_to_worst)
            }
        }

        function high_yield(dailyTradeArr) {
            const accumulated = dailyTradeArr.reduce(accumulate_high_yield_ignore_null, Number.NEGATIVE_INFINITY)    // The Number.MIN_VALUE static data property represents the smallest positive numeric value representable in JavaScript
            if (accumulated === Number.NEGATIVE_INFINITY) {
                return NO_YIELDS_REPORTED
            } else {
                return accumulated
            }
        }

        function accumulate_low_yield_ignore_null(accumulator, trade) {
            if (trade.yield_to_worst === null) {
                return accumulator
            } else {
                return Math.min(accumulator, trade.yield_to_worst)
            }
        }

        function low_yield(dailyTradeArr) {
            const accumulated = dailyTradeArr.reduce(accumulate_low_yield_ignore_null, Number.MAX_VALUE)
            if (accumulated === Number.MAX_VALUE) {
                return NO_YIELDS_REPORTED
            } else {
                return accumulated
            }
        }

        function accumulate_high_dollar_price_ignore_null(accumulator, trade) {
            if (trade.dollar_price === null) {
                return accumulator
            } else {
                return Math.max(accumulator, trade.dollar_price)
            }
        }

        function high_price(dailyTradeArr) {
            const accumulated = dailyTradeArr.reduce(accumulate_high_dollar_price_ignore_null, Number.NEGATIVE_INFINITY)    // The Number.MIN_VALUE static data property represents the smallest positive numeric value representable in JavaScript
            if (accumulated === Number.NEGATIVE_INFINITY) {
                return NO_DOLLAR_PRICES_REPORTED
            } else {
                return accumulated
            }
        }

        function accumulate_low_dollar_price_ignore_null(accumulator, trade) {
            if (trade.dollar_price === null) {
                return accumulator
            } else {
                return Math.min(accumulator, trade.dollar_price)
            }
        }

        function low_price(dailyTradeArr) {
            const accumulated = dailyTradeArr.reduce(accumulate_low_dollar_price_ignore_null, Number.MAX_VALUE)
            if (accumulated === Number.MAX_VALUE) {
                return NO_DOLLAR_PRICES_REPORTED
            } else {
                return accumulated
            }
        }

        // function avg_spread(dailyTradeArr) {
        //     const total_spread = dailyTradeArr.reduce((accumulator, trade) => accumulator + trade.yield_spread, 0)
        //     return total_spread / dailyTradeArr.length
        // }

        function replaceNaNWithZero(val) {
            if (isNaN(val)) {
                return 0
            }
            return val
        }

        function calculateTotalByTradeType(tradeData) {
            return tradeData.reduce((acc, trade) => {
              if (!acc[trade.trade_type]) {
                acc[trade.trade_type] = 0
              } 
              acc[trade.trade_type] += trade.size
              return acc
            }, {})
        }

        const grouped = groupBy(curve_history_data, 'date')
        
        var summary_by_date = []
        for (var trade_date in grouped) {
            var summary_by_date_entry = {}
            summary_by_date_entry['showDetails'] = false
            summary_by_date_entry['date'] = trade_date
            var rowId = 'k' + trade_date.replace('/','_').replace('/','_')
            summary_by_date_entry['id'] = rowId
            summary_by_date_entry['total'] = total(grouped[trade_date])
            summary_by_date_entry['count'] = count(grouped[trade_date])
            summary_by_date_entry['high_yield'] = high_yield(grouped[trade_date])
            summary_by_date_entry['low_yield'] = low_yield(grouped[trade_date])
            summary_by_date_entry['high_price'] = high_price(grouped[trade_date])
            summary_by_date_entry['low_price'] = low_price(grouped[trade_date])
            // summary_by_date_entry['avg_spread'] = avg_spread(grouped[trade_date])

            var volByTradeType = calculateTotalByTradeType(grouped[trade_date])
            summary_by_date_entry['ddVol'] = replaceNaNWithZero(volByTradeType['D'])
            summary_by_date_entry['dpVol'] = replaceNaNWithZero(volByTradeType['P'])
            summary_by_date_entry['dsVol'] = replaceNaNWithZero(volByTradeType['S'])
            // summary_by_date_entry['net'] = replaceNaNWithZero(volByTradeType['P']) - replaceNaNWithZero(volByTradeType['S'])
            // setSummary_by_date(oldArray => [...oldArray, summary_by_date_entry])
            summary_by_date.push(summary_by_date_entry)
        }
        
        if (true) {    // need to check that curve_history_data has data in it at some point
            return(summary_by_date.map((c) => (
                <>
                <tr key={c.id} data-id={c.id} onClick={handleClick} style={hoverStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
                    <td>{c.date}</td>
                    <td>{c.total / 1000}</td>
                    <td>{c.count}</td>
                    {/* <td>{Math.round(c.high_spread, 0)}</td> */}
                    {/* <td>{Math.round(c.low_spread, 0)}</td> */}
                    {/* <td>{Math.round(c.avg_spread, 0)}</td> */}
                    <td>{c.high_price === NO_DOLLAR_PRICES_REPORTED ? '' : c.high_price + ' / '}{c.low_price}</td>
                    <td>{c.high_yield === NO_YIELDS_REPORTED ? '' : c.high_yield + ' / '}{c.low_yield}</td>
                    <td>{c.dpVol / 1000}</td>
                    <td>{c.dsVol / 1000}</td>
                    {/* <td>{c.net / 1000}</td> */}
                    <td>{c.ddVol / 1000}</td>                    
                </tr>
                
                {hiddenRows.includes(c.id) && (
                <tr key={c.id + '_hidden'}>
                    <td colspan='10'>
                    <Table striped bordered>
                        <thead>
                            <tr>
                                <th>Trade Date & Time</th>
                                {/* <th>Spread (bps)</th> */}
                                {/* <th>ficc Curve (bps)</th> */}
                                <th>Yield (%)</th>
                                <th>Price (%)</th>
                                <th>Yield to Worst Date</th>
                                <th>Trade Amount (k)</th>
                                <th>Trade Type</th>
                            </tr>
                        </thead>
                    <tbody>{dispSingleTradeDetails(grouped, c.date)}</tbody>
                    </Table>
                    </td>
                </tr>)}
            </>
                ))
            )
        }
    }
  
    if (curve_history_data.length !== 0) {
        return (
            <div>
                <br></br>
                <h5>Recent trade history for {cusipForDisplay}; click a row to expand daily trades</h5> {/*    (up to 32 trades)*/}
                <Table striped bordered>
                <thead>
                    <tr>
                        <th>Trade Date</th>
                        <th>Vol (k)</th>
                        <th>Trade Count</th>
                        <th>High / Low Price (%)</th>
                        <th>High / Low Yield (%)</th>
                        {/* <th>Low Spread (bps)</th> */}
                        {/* <th>Avg Spread (bps)</th> */}
                        <th>Purchase from Customer Vol (k)</th>
                        <th>Sale to Customer Vol (k)</th>
                        {/* <th>Net (k)</th> */}
                        <th>Inter-dealer Vol (k)</th>
                    </tr>
                </thead>
                <tbody>{dispTradesStats()}</tbody>
                </Table>
            </div>
        )
    }
}

export default (FiccCurveHistory)