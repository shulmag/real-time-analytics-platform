/*
 * @Date: 2024-04-25
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadComplianceFile } from '../services/priceService';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import styles from './Compliance.module.css';
import FONT_SIZE from './pricing/globalVariables';
import NavBarTop from './navBarTop';
import { Container, Row, Col, Card, Spinner, Button, Form, InputGroup, Tab, Tabs, Table } from 'react-bootstrap';
import { useTable, usePagination, useSortBy } from 'react-table';
import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';

function getComplianceRatingClass(rating) {
    const ratingClasses = {
        'Great': styles.ratingGreat,
        'Good': styles.ratingGood,
        'Fair': styles.ratingFair,
        'Poor': styles.ratingPoor,
    };
    return ratingClasses[rating] || '';
}

function Compliance() {
    const [key, setKey] = useState('compliance');
    const [userEmail, setUserEmail] = useState('');
    const [isBatchProcessing, setIsBatchProcessing] = useState(false);
    const [isDownloadProcessing, setIsDownloadProcessing] = useState(false);
    const [file, setFile] = useState();
    const [tableData, setTableData] = useState([]);
    const [showTable, setShowTable] = useState(false);
    const loggedOutMessage = 'You have been logged out due to a period of inactivity. Refresh the page!';
    const nav = useNavigate();

    const tradeType = [
        { key: 'D', text: 'Inter-Dealer' },
        { key: 'P', text: 'Bid Side' },
        { key: 'S', text: 'Offered Side' },
    ];

    const [batchValues, setBatchValues] = useState({
        quantity: 500,
        tradeType: 'S',
    });

    const resultsPerPage = 25; // Number of results to show per page

    const columns = React.useMemo(
        () => [
            { Header: 'CUSIP', accessor: 'cusip' },
            { 
                Header: 'Quantity', 
                accessor: 'quantity',
                Cell: ({ value }) => value.toLocaleString()    // format numbers with commas
            },
            { Header: 'Trade Type', accessor: 'trade_type' },
            { Header: 'YTW', accessor: 'ytw' },
            { Header: 'Price', accessor: 'price' },
            { Header: 'User Price', accessor: 'user_price' },
            { Header: 'Bid Ask Price Delta', accessor: 'bid_ask_price_delta' },
            { Header: 'Yield to Worst Date', accessor: 'yield_to_worst_date' },
            { Header: 'Coupon', accessor: 'coupon' },
            { Header: 'Security Description', accessor: 'security_description' },
            { Header: 'Maturity Date', accessor: 'maturity_date' },
            { Header: 'Trade Datetime', accessor: 'trade_datetime' },
            { Header: 'Notes', accessor: 'error_message' },
            {
                Header: 'Compliance Rating',
                accessor: 'compliance_rating',
                Cell: ({ value }) => (
                    <span className={getComplianceRatingClass(value)}>{value}</span>
                ),
                sortType: (a, b) => {
                    const order = ['Great', 'Good', 'Fair', 'Poor'];
                    return order.indexOf(a.values.compliance_rating) - order.indexOf(b.values.compliance_rating);
                },
            },
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
        setPageSize,
        state: { pageIndex, pageSize },
    } = useTable(
        {
            columns,
            data: tableData,
            initialState: { pageIndex: 0, pageSize: resultsPerPage },
        },
        useSortBy,
        usePagination
    );

    function handleChange(event) {
        setFile(event.target.files[0]);
    }

    async function getAuthenticationToken() {
        const auth = getAuth();
        const user = auth.currentUser;
        try {
            const token = await user.getIdToken(true);
            setUserEmail(user.email);
            return token;
        } catch (error) {
            console.log(error);
            throw new Error(loggedOutMessage);
        }
    }

    async function uploadFile(isDownload) {
        if (typeof file === 'undefined') {
            alert('No file was uploaded');
            return;
        }

        isDownload ? setIsDownloadProcessing(true) : setIsBatchProcessing(true);
        const token = await getAuthenticationToken();

        const formData = new FormData();
        formData.append('file', file);
        formData.append('access_token', token);
        formData.append('amount', batchValues['quantity']);
        formData.append('tradeType', batchValues['tradeType']);
        if (isDownload) { 
            formData.append('download', true);
            formData.append('useCachedPricedFile', true)
        }

        try {
            const response = await uploadComplianceFile(formData);
            if (isDownload) {
                const href = URL.createObjectURL(response.data);
                const link = document.createElement('a');
                link.href = href;
                link.setAttribute('download', 'preds.csv');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(href);
            } else {
                const text = await response.data.text();
                const data = JSON.parse(text);
                const jsonData = JSON.parse(data);
                const dataArray = Object.keys(jsonData.cusip).map((key) => ({
                    cusip: jsonData.cusip[key],
                    quantity: jsonData.quantity[key],
                    trade_type: jsonData.trade_type[key],
                    ytw: jsonData.ytw[key],
                    price: jsonData.price[key],
                    user_price: jsonData.user_price[key],
                    bid_ask_price_delta: jsonData.bid_ask_price_delta[key],
                    yield_to_worst_date: jsonData.yield_to_worst_date[key],
                    coupon: jsonData.coupon[key],
                    security_description: jsonData.security_description[key],
                    maturity_date: jsonData.maturity_date[key],
                    trade_datetime: jsonData.trade_datetime[key],
                    error_message: jsonData.error_message[key],
                    compliance_rating: jsonData.compliance_rating[key],
                }));
                setTableData(dataArray);
                setShowTable(true);
            }
        } catch (error) {
            alert('Compliance Module Error: ' + error.message);
        } finally {
            isDownload ? setIsDownloadProcessing(false) : setIsBatchProcessing(false);
        }
    }

    function onFileUpload(event) {
        event.preventDefault();
        uploadFile(true);
    }

    function jsonOnFileUpload(event) {
        event.preventDefault();
        setShowTable(false);
        uploadFile(false);
    }

    function setBatch(name) {
        return function ({ target: { value } }) {
            setBatchValues((oldBatchValues) => ({ ...oldBatchValues, [name]: value }));
        };
    }

    useEffect(() => {
        const auth = getAuth();

        onAuthStateChanged(auth, (user) => {
            if (user) {
                user.getIdToken(true).then((token) => {
                    setUserEmail(user.email);
                });
            } else {
                nav('/login');
            }
        });
    }, [nav]);

    return (
        <Container fluid className='flex justify-content-center' style={{ fontSize: FONT_SIZE }}>
            <div>
                <NavBarTop userEmail={userEmail} />
                <Tabs id='controlled-tabs' activeKey={key} onSelect={(k) => setKey(k)} className='mb-3'>
                    <Tab eventKey='compliance' title='Compliance'>
                        <Card xs='auto'>
                            <Card.Body>
                                <div>
                                    <Form onSubmit={jsonOnFileUpload}>
                                        {/* Search Form */}
                                        <Form.Group controlId='formFile' className='mb-3'>
                                            <Row>
                                                <Col>
                                                    <Form.Label className='font-weight-light' size='sm'>CSV File: </Form.Label>
                                                    <Form.Control type='file' onChange={handleChange} />
                                                </Col>
                                                <Col>
                                                    <Form.Group className='mb-3'>
                                                        <Form.Label className='font-weight-light' size='sm'>Trade Amount (thousands): </Form.Label>
                                                        <InputGroup>
                                                            <InputGroup.Text>$ (k)</InputGroup.Text>
                                                            <Form.Control placeholder='Dollar Amount' type='number' required min='5' max='10000' name='amount' value={batchValues.quantity} onChange={setBatch('quantity')} />
                                                        </InputGroup>
                                                    </Form.Group>
                                                </Col>
                                                <Col>
                                                    <Form.Group className='mb-3'>
                                                        <Form.Label className='font-weight-light' size='sm'>Trade Type: </Form.Label>
                                                        <Form.Select required name='tradeType' value={batchValues.tradeType} onChange={setBatch('tradeType')}>
                                                            {tradeType.map((o) => (
                                                                <option key={o.key} value={o.key}>
                                                                    {o.text}
                                                                </option>
                                                            ))}
                                                        </Form.Select>
                                                    </Form.Group>
                                                </Col>
                                                <Col className='d-flex align-items-center'>
                                                    <Form.Group className='mb-3'>
                                                    <Form.Label className='font-weight-light' style={{ color: 'white' }} size='sm'>.</Form.Label>
                                                       <br></br>
                                                    {isBatchProcessing ? (
                                                        <Button className='btn btn-primary submitButtonMargin' disabled>
                                                        <Spinner as='span' animation='border' size='sm' role='status' aria-hidden='true'/> Processing...
                                                        </Button>
                                                    ) : (
                                                        <Button className='btn btn-primary submitButtonMargin' type='submit'>
                                                            Submit
                                                        </Button>
                                                    )}
                                                    </Form.Group>
                                                </Col>
                                            </Row>
                                        </Form.Group>
                                    </Form>
                                    {!showTable && (
                                        <div>
                                            <Form.Label>
                                                Here's what the CSV should look like (skip the header):
                                            </Form.Label>
                                            <Table striped bordered size='sm' className='mt-2'>
                                                <thead>
                                                    <tr>
                                                        <th>CUSIP</th>
                                                        <th>Trade Amount (in thousands)</th>
                                                        <th>Trade Type</th>
                                                        <th>Trade Price</th>
                                                        <th>Trade Datetime</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    <tr>
                                                        <td>64971XQM3</td>
                                                        <td>100</td>
                                                        <td>P</td>
                                                        <td>99.500</td>
                                                        <td>2024-10-03 12:52:48</td>
                                                    </tr>
                                                    <tr>
                                                        <td>950885SN4</td>
                                                        <td>250</td>
                                                        <td>S</td>
                                                        <td>101.250</td>
                                                        <td>2024-10-03 14:00:00</td>
                                                    </tr>
                                                </tbody>
                                            </Table>
                                            <Form.Text className='text-muted'>
                                                If no trade amount is entered, the default value from the box above will be used (bounded between 5 and 10000). For trade type, use 'P' for Bid Side, 'S' for Offered Side, or 'D' for Inter-Dealer. If no trade type is entered, the default value from the box above will be used. The trade datetime can also be left empty, and in this case, the trade will be priced in realtime, and will provide results a lot faster than specifying a trade datetime.
                                            </Form.Text>
                                        </div>
                                    )}
                                    {showTable && (
                                                <div>
                                                    <h5>Results</h5>
                                                    {tableData.length > 0 && (
                                                        <>
                                                            <div className='d-flex justify-content-end mb-2'>
                                                                {isDownloadProcessing ? (
                                                                    <Button className='btn btn-success' disabled>
                                                                        Downloading...
                                                                    </Button>
                                                                ) : (
                                                                    <Button className='btn btn-success' onClick={onFileUpload}>
                                                                        Download CSV
                                                                    </Button>
                                                                )}
                                                            </div>
                                                            <Table striped bordered hover size='sm' responsive className='mt-2' {...getTableProps()}>
                                                            <thead className='thead-dark'>
                                                                {headerGroups.map((headerGroup) => (
                                                                    <tr {...headerGroup.getHeaderGroupProps()}>
                                                                        {headerGroup.headers.map((column) => (
                                                                            <th {...column.getHeaderProps(column.getSortByToggleProps())} style={{ cursor: 'pointer' }}>
                                                                                <div className='d-flex align-items-center'>
                                                                                    {column.render('Header')}
                                                                                    <span className='ms-1'>
                                                                                        {column.isSorted ? (
                                                                                            column.isSortedDesc ? (
                                                                                                <FaSortDown />
                                                                                            ) : (
                                                                                                <FaSortUp />
                                                                                            )
                                                                                        ) : (
                                                                                            <FaSort />
                                                                                        )}
                                                                                    </span>
                                                                                </div>
                                                                            </th>
                                                                        ))}
                                                                    </tr>
                                                                ))}
                                                            </thead>
                                                                <tbody {...getTableBodyProps()}>
                                                                    {page.map((row) => {
                                                                        prepareRow(row);
                                                                        return (
                                                                            <tr {...row.getRowProps()}>
                                                                                {row.cells.map((cell) => (
                                                                                    <td {...cell.getCellProps()} className={cell.column.id === 'compliance_rating' ? getComplianceRatingClass(cell.value) : ''}>
                                                                                        {cell.render('Cell')}
                                                                                    </td>
                                                                                ))}
                                                                            </tr>
                                                                        );
                                                                    })}
                                                                </tbody>
                                                            </Table>
                                                            <div className='pagination d-flex justify-content-center align-items-center mt-3'>
                                                                <Button onClick={() => gotoPage(0)} disabled={!canPreviousPage} className='me-2'>
                                                                    {'<<'}
                                                                </Button>
                                                                <Button onClick={() => previousPage()} disabled={!canPreviousPage} className='me-2'>
                                                                    {'<'}
                                                                </Button>
                                                                <span className='me-2'>
                                                                    Page{' '}
                                                                    <strong>
                                                                        {pageIndex + 1} of {pageOptions.length}
                                                                    </strong>
                                                                </span>
                                                                <Button onClick={() => nextPage()} disabled={!canNextPage} className='me-2'>
                                                                    {'>'}
                                                                </Button>
                                                                <Button onClick={() => gotoPage(pageCount - 1)} disabled={!canNextPage} className='me-2'>
                                                                    {'>>'}
                                                                </Button>
                                                                <div className='d-flex align-items-center'>
                                                                    <span className='me-2'>Go to page:</span>
                                                                    <input
                                                                        type='number'
                                                                        defaultValue={pageIndex + 1}
                                                                        onChange={(e) => {
                                                                            const page = e.target.value ? Number(e.target.value) - 1 : 0;
                                                                            gotoPage(page);
                                                                        }}
                                                                        style={{ width: '60px' }}
                                                                        className='form-control form-control-sm me-2'
                                                                    />
                                                                </div>
                                                                <div className='d-flex align-items-center'>
                                                                    <span className='me-2'>Show:</span>
                                                                    <select
                                                                        value={pageSize}
                                                                        onChange={(e) => {
                                                                            setPageSize(Number(e.target.value));
                                                                        }}
                                                                        className='form-select form-select-sm'
                                                                    >
                                                                        {[10, 25, 50, 100, 250].map((pageSize) => (
                                                                            <option key={pageSize} value={pageSize}>
                                                                                {pageSize}
                                                                            </option>
                                                                        ))}
                                                                    </select>
                                                                </div>
                                                            </div>
                                                        </>
                                                    )}
                                                </div>
                                            )}
                                </div>
                            </Card.Body>
                        </Card>
                    </Tab>
                </Tabs>
            </div>
        </Container>
    );
}

export default Compliance;