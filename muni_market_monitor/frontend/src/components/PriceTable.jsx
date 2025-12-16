/**
 * Description: Component for displaying municipal bond price data.
 * Shows real-time prices and compliance (4 PM) prices with comparisons.
 */

import React, { useEffect, useState } from 'react';
import { Table, Container, Alert, Spinner, Button } from 'react-bootstrap';
import { getPrices } from '../services/api';
import { useTable, useSortBy, usePagination } from 'react-table';
import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';

function PriceTable() {
  const [prices, setPrices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [pageSize, setLocalPageSize] = useState(50);

  const getBackgroundClass = (delta) => {
    if (delta > 0) {
      return 'bg-success-subtle';
    } else if (delta < 0) {
      return 'bg-danger-subtle';
    }
    return '';
  };

  const columns = React.useMemo(
    () => [
      { Header: 'CUSIP', accessor: 'cusip' },
      { 
        Header: 'Amount (1000s)', 
        accessor: 'quantity',
        Cell: ({ value }) => (value / 1000).toFixed(0)
      },
      { Header: 'Trade Type', accessor: 'trade_type' },
      { 
        Header: 'Price (RT)', 
        accessor: 'price_realtime',
        Cell: ({ value, row }) => value.toFixed(3),
        cellClassName: row => 
          row.original.price_delta > 0 ? 'bg-success-subtle' : 
          row.original.price_delta < 0 ? 'bg-danger-subtle' : ''
      },
      { 
        Header: 'Price (Yday)', 
        accessor: 'price_yesterday',
        Cell: ({ value }) => value.toFixed(3)
      },
      { 
        Header: 'Price Δ', 
        accessor: 'price_delta',
        Cell: ({ value }) => renderDelta(value)
      },
      { 
        Header: 'YTW (RT)', 
        accessor: 'ytw_realtime',
        Cell: ({ value }) => value.toFixed(3),
        cellClassName: row => 
          row.original.ytw_delta > 0 ? 'bg-success-subtle' : 
          row.original.ytw_delta < 0 ? 'bg-danger-subtle' : ''
      },
      { 
        Header: 'YTW (Yday)', 
        accessor: 'ytw_yesterday',
        Cell: ({ value }) => value.toFixed(3)
      },
      { 
        Header: 'YTW Δ', 
        accessor: 'ytw_delta',
        Cell: ({ value }) => renderDelta(value)
      },
      { Header: 'Coupon', accessor: 'coupon' },
      { Header: 'Description', accessor: 'security_description' }
    ],
    []
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    page,
    prepareRow,
    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    setPageSize: setReactTablePageSize,
    state: { pageIndex }
  } = useTable(
    {
      columns,
      data: prices,
      initialState: { 
        pageIndex: 0, 
        pageSize: 50
      }
    },
    useSortBy,
    usePagination
  );

  const fetchPrices = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getPrices();
      setPrices(data);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error fetching prices:', error);
      setError('Failed to load prices. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrices();
  }, []);

  const renderDelta = (delta) => {
    if (delta > 0) {
      return <span className="text-success">▲ {delta.toFixed(3)}</span>;
    } else if (delta < 0) {
      return <span className="text-danger">▼ {delta.toFixed(3)}</span>;
    }
    return delta.toFixed(3);
  };

  const handlePageSizeChange = (newSize) => {
    const size = Number(newSize);
    setLocalPageSize(size);
    setReactTablePageSize(size);
  };

  if (loading) {
    return (
      <Container fluid>
        <div className="text-center py-5">
          <Spinner animation="border" variant="primary" />
          <div className="mt-3">
            <h5 className="text-muted">Loading Price Data...</h5>
            <small className="text-muted">
              Fetching real-time and compliance prices
            </small>
          </div>
        </div>
      </Container>
    );
  }

  if (error) {
    return (
      <Container fluid>
        <Alert variant="danger" className="mt-3">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container fluid>
      <div className="d-flex justify-content-end align-items-center mb-3 mt-3">
        <Button 
          onClick={fetchPrices} 
          variant="primary"
          size="sm"
          className="me-2"
          disabled={loading}
        >
          <i className="bi bi-arrow-clockwise me-1"></i>
          Refresh
        </Button>
        {lastUpdated && (
          <small className="text-muted">
            Last updated: {lastUpdated}
          </small>
        )}
      </div>

      <Table 
        responsive 
        hover 
        striped 
        bordered 
        className="shadow-sm"
        {...getTableProps()}
      >
        <thead className="table-light">
          {headerGroups.map(headerGroup => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map(column => (
                <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                  <div className="d-flex align-items-center">
                    {column.render('Header')}
                    <span className="ms-1">
                      {column.isSorted ? (
                        column.isSortedDesc ? <FaSortDown /> : <FaSortUp />
                      ) : <FaSort />}
                    </span>
                  </div>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody {...getTableBodyProps()}>
          {page.map(row => {
            prepareRow(row);
            return (
              <tr {...row.getRowProps()}>
                {row.cells.map(cell => (
                  <td 
                    {...cell.getCellProps()}
                    className={cell.column.cellClassName ? cell.column.cellClassName(row) : ''}
                  >
                    {cell.render('Cell')}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </Table>

      <div className="pagination d-flex justify-content-center align-items-center mt-3">
        <Button onClick={() => gotoPage(0)} disabled={!canPreviousPage} className="me-2">{'<<'}</Button>
        <Button onClick={() => previousPage()} disabled={!canPreviousPage} className="me-2">{'<'}</Button>
        <span className="me-2">
          Page <strong>{pageIndex + 1} of {pageOptions.length}</strong>
        </span>
        <Button onClick={() => nextPage()} disabled={!canNextPage} className="me-2">{'>'}</Button>
        <Button onClick={() => gotoPage(pageCount - 1)} disabled={!canNextPage} className="me-2">{'>>'}</Button>
        
        <div className="d-flex align-items-center ms-2">
          <span className="me-2">Show:</span>
          <select
            value={pageSize}
            onChange={e => handlePageSizeChange(e.target.value)}
            className="form-select form-select-sm"
          >
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={250}>250</option>
          </select>
        </div>
      </div>
    </Container>
  );
}

export default PriceTable;
