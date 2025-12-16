/*
 * @Date: 2021-09-29 13:05:11 
 */
import React from "react";
import Table from "react-bootstrap/Table"
import { Link } from "react-router-dom";

function CusipTable({cusip_table_data,expand}) {
  function CusipTableData() {
    if(cusip_table_data !== 'no token'){
      return (
        cusip_table_data.map((c) => (
          <tr>
            <td id={c.cusip}><Link style={{color:"#59e4fd"}} to="#" onClick={() => expand(c.cusip)}> {c.cusip}</Link></td>
            <td>{c.count}</td>
            <td>{c.security_description}</td>
            <td>{c.coupon}</td>
            <td>{c.ytw}</td>
            <td>{c.price}</td>
            <td style={{whiteSpace:"nowrap"}}>{c.calculation_method}</td>            
          </tr>
        ))
      );}
    };

    return(
      <div>
        <Table style={{color:"#e0e1fe",backgroundColor:"#303030",borderColor:"606060"}} striped bordered>
          <thead>
            <tr>
              <th>CUSIP</th>
              <th># of Trades</th>
              <th>Description</th>
              <th>Coupon (%)</th>
              <th>Est. Yield (%)</th>
              <th>Est. Price ($)</th>
              <th>Calc Date</th>
            </tr>
          </thead>
          <tbody>{CusipTableData()}</tbody>
        </Table>
      </div>
    );
};

export default CusipTable;