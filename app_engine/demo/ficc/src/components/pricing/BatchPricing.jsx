import React from 'react';
import { 
  Form, 
  Row, 
  Col, 
  Button, 
  Spinner, 
  Table,
  InputGroup,
  Card
} from 'react-bootstrap';
import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';

function BatchPricing({
  handleDisplay,
  handleChange,
  handleDownload,
  file,
  batchValues,
  setBatch,
  isBatchProcessing,
  isDownloadProcessing,
  showTable,
  tableData,
  tableProps,
  isPricing,
  tradeType
}) {
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
    setPageSize,
    state: { pageIndex, pageSize },
  } = tableProps;

  return (
    // Use the same card class and styling
    <Card className="ficc-card">
      <Card.Body className="ficc-card-body">
        <Form onSubmit={handleDisplay}>
          <Form.Group controlId="formFile" className="mb-3">
            <Row>
              <Col>
                <Form.Label 
                  className="text-secondary mb-3" 
                  style={{ fontSize: '0.95rem', lineHeight: '1.5' }}
                >
                  Upload a CSV file with a list of CUSIPs (each on a separate row). 
                  Optionally, enter a trade amount (in thousands) in the second column 
                  and a trade type in the third column.
                </Form.Label>
              </Col>
            </Row>
            
            <Row className="align-items-end mb-3">
              <Col md={3}>
                <Form.Group className="mb-0">
                  <Form.Label 
                    className="text-secondary mb-2" 
                    style={{ fontSize: '0.9rem', fontWeight: '500' }}
                  >
                    CSV File
                  </Form.Label>
                  <Form.Control 
                    type="file" 
                    onChange={handleChange} 
                    className="shadow-sm"
                    style={{
                      borderRadius: '6px',
                      border: '1px solid #e2e8f0',
                      padding: '8px 12px'
                    }}
                  />
                </Form.Group>
              </Col>

              <Col md={3}>
                <Form.Group className="mb-0">
                  <Form.Label 
                    className="text-secondary mb-2" 
                    style={{ fontSize: '0.9rem', fontWeight: '500' }}
                  >
                    Trade Amount (thousands)
                  </Form.Label>
                  <InputGroup>    
                    <InputGroup.Text
                      style={{
                        backgroundColor: '#f8fafc',
                        border: '1px solid #e2e8f0',
                        borderRight: 'none'
                      }}
                    >
                      $ (k)
                    </InputGroup.Text>
                    <Form.Control 
                      placeholder="Dollar Amount" 
                      type="number" 
                      required 
                      min="5" 
                      max="10000" 
                      name="amount" 
                      value={batchValues.quantity} 
                      onChange={setBatch('quantity')}
                      style={{
                        borderRadius: '0 6px 6px 0',
                        padding: '10px 12px',
                        border: '1px solid #e2e8f0',
                        fontSize: '0.95rem',
                        backgroundColor: isPricing ? '#f7fafc' : 'white',
                        cursor: isPricing ? 'not-allowed' : 'text'
                      }}
                      disabled={isPricing}
                      className="shadow-sm"
                    />
                  </InputGroup>
                </Form.Group>
              </Col>

              <Col md={3}>
                <Form.Group className="mb-0">
                  <Form.Label 
                    className="text-secondary mb-2" 
                    style={{ fontSize: '0.9rem', fontWeight: '500' }}
                  >
                    Trade Type
                  </Form.Label>
                  <Form.Select 
                    required 
                    name="tradeType" 
                    value={batchValues.tradeType} 
                    onChange={setBatch('tradeType')}
                    style={{
                      borderRadius: '6px',
                      padding: '10px 12px',
                      border: '1px solid #e2e8f0',
                      fontSize: '0.95rem',
                      backgroundColor: isPricing ? '#f7fafc' : 'white',
                      cursor: isPricing ? 'not-allowed' : 'pointer'
                    }}
                    disabled={isPricing}
                    className="shadow-sm"
                  >
                    {tradeType.map((o) => (
                      <option key={o.key} value={o.key}>
                        {o.text}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>
              </Col>

              <Col md="auto">
                {isBatchProcessing ? (
                  <Button 
                    disabled
                    style={{
                      padding: '10px 20px',
                      backgroundColor: '#3182ce',
                      border: 'none',
                      borderRadius: '6px'
                    }}
                  >
                    <Spinner 
                      as="span" 
                      animation="border" 
                      size="sm" 
                      role="status" 
                      aria-hidden="true"
                      className="me-2"
                    /> 
                    Processing...
                  </Button>
                ) : (
                  <Button 
                    type="submit"
                    style={{
                      padding: '10px 20px',
                      backgroundColor: '#3182ce',
                      border: 'none',
                      borderRadius: '6px',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#2c5282'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#3182ce'}
                  >
                    Submit
                  </Button>
                )}
              </Col>
            </Row>
          </Form.Group>
        </Form>

        {!showTable && (
          <div className="mt-4">
            <h5 
              className="text-secondary mb-3" 
              style={{ fontSize: '1.1rem', fontWeight: '600', color: '#2d3748' }}
            >
              CSV Formatting Instructions:
            </h5>
            <div className="table-responsive">
              <Table className="ficc-table" striped bordered size="sm">
                <thead>
                  <tr>
                    <th style={{ backgroundColor: '#edf2f7', color: '#4a5568', fontWeight: '600', fontSize: '0.9rem', padding: '12px 16px' }}>
                      CUSIP
                    </th>
                    <th style={{ backgroundColor: '#edf2f7', color: '#4a5568', fontWeight: '600', fontSize: '0.9rem', padding: '12px 16px' }}>
                      Trade Amount (in thousands)
                    </th>
                    <th style={{ backgroundColor: '#edf2f7', color: '#4a5568', fontWeight: '600', fontSize: '0.9rem', padding: '12px 16px' }}>
                      Trade Type
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>801315LS9</td>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>20</td>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>P</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>228130GF1</td>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>50</td>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>D</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>751622RN3</td>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>20</td>
                    <td style={{ padding: '10px 16px', fontSize: '0.9rem' }}>S</td>
                  </tr>
                  {/* ...etc... */}
                </tbody>
              </Table>
            </div>
            <p className="mt-3 text-muted" style={{ fontSize: '0.9rem' }}>
              If no trade amount is entered, the default value is used (bounded between 5 and 10000). 
              For trade type, use 'P' for Bid Side, 'S' for Offered Side, or 'D' for Inter-Dealer. 
              If no trade type is entered, the default is used.
            </p>
          </div>
        )}

        {showTable && tableData.length > 0 && (
          <div className="mt-4">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <h5 
                style={{ fontSize: '1.1rem', fontWeight: '600', color: '#2d3748', margin: 0 }}
              >
                Results
              </h5>
              <div>
                {isDownloadProcessing ? (
                  <Button 
                    className="btn" 
                    disabled
                    style={{
                      backgroundColor: '#48bb78',
                      border: 'none',
                      borderRadius: '6px',
                      padding: '8px 16px',
                      fontSize: '0.9rem'
                    }}
                  >
                    <Spinner 
                      as="span" 
                      animation="border" 
                      size="sm" 
                      role="status" 
                      aria-hidden="true"
                      className="me-2"
                    />
                    Downloading...
                  </Button>
                ) : (
                  <Button 
                    className="btn" 
                    onClick={handleDownload}
                    style={{
                      backgroundColor: '#48bb78',
                      border: 'none',
                      borderRadius: '6px',
                      padding: '8px 16px',
                      fontSize: '0.9rem',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#38a169'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#48bb78'}
                  >
                    Download CSV
                  </Button>
                )}
              </div>
            </div>
            
            <div className="table-responsive">
              <Table 
                striped 
                bordered 
                hover 
                size="sm" 
                className="ficc-table" 
                {...getTableProps()}
              >
                <thead>
                  {headerGroups.map((headerGroup, i) => (
                    <tr {...headerGroup.getHeaderGroupProps()} key={i}>
                      {headerGroup.headers.map((column, j) => (
                        <th 
                          {...column.getHeaderProps(column.getSortByToggleProps())} 
                          key={j}
                          style={{ 
                            cursor: 'pointer',
                            backgroundColor: '#edf2f7',
                            color: '#4a5568',
                            fontWeight: '600',
                            fontSize: '0.9rem',
                            padding: '12px 16px' 
                          }}
                        >
                          <div className="d-flex align-items-center">
                            {column.render('Header')}
                            <span className="ms-1">
                              {column.isSorted ? (
                                column.isSortedDesc ? (
                                  <FaSortDown />
                                ) : (
                                  <FaSortUp />
                                )
                              ) : (
                                <FaSort style={{ opacity: 0.4 }} />
                              )}
                            </span>
                          </div>
                        </th>
                      ))}
                    </tr>
                  ))}
                </thead>
                <tbody {...getTableBodyProps()}>
                  {page.map((row, i) => {
                    prepareRow(row);
                    return (
                      <tr {...row.getRowProps()} key={i}>
                        {row.cells.map((cell, j) => (
                          <td 
                            {...cell.getCellProps()} 
                            key={j} 
                            style={{ 
                              fontSize: '0.9rem', 
                              padding: '10px 16px'
                            }}
                          >
                            {cell.render('Cell')}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </Table>
            </div>

            <div className="d-flex justify-content-between align-items-center mt-3">
              <div>
                Page{' '}
                <strong>
                  {pageIndex + 1} of {pageOptions.length}
                </strong>
              </div>
              <div>
                <Button 
                  variant="outline-secondary" 
                  size="sm" 
                  onClick={() => gotoPage(0)} 
                  disabled={!canPreviousPage}
                  style={{ marginRight: '5px' }}
                >
                  {'<<'}
                </Button>
                <Button 
                  variant="outline-secondary" 
                  size="sm" 
                  onClick={() => previousPage()} 
                  disabled={!canPreviousPage}
                  style={{ marginRight: '5px' }}
                >
                  {'<'}
                </Button>
                <Button 
                  variant="outline-secondary" 
                  size="sm" 
                  onClick={() => nextPage()} 
                  disabled={!canNextPage}
                  style={{ marginRight: '5px' }}
                >
                  {'>'}
                </Button>
                <Button 
                  variant="outline-secondary" 
                  size="sm" 
                  onClick={() => gotoPage(pageCount - 1)} 
                  disabled={!canNextPage}
                >
                  {'>>'}
                </Button>
              </div>
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}

export default BatchPricing;
