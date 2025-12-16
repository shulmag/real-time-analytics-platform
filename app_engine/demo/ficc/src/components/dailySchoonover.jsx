/*
 * @Date: 2022-09-28 
 */

import React from 'react'
import Table from 'react-bootstrap/Table'

import { useState, useEffect} from 'react'
import { useNavigate } from 'react-router-dom'
import { getDailySchoonover } from '../services/priceService'

import NavBarTop from './navBarTop'
import { getAuth, onAuthStateChanged } from 'firebase/auth'
import Placeholder from 'react-bootstrap/Placeholder'

import Container from 'react-bootstrap/Container'
import moment from 'moment-timezone'

import FONT_SIZE from './pricing/globalVariables'


function DailySchoonover(props) {
	var fbToken = ''
	const nav = useNavigate()

	const [daily_schoonover_data,setMarketData] = React.useState('')
	const [stateToken,setStateToken] = useState('')
	const [isLoading, setLoading] = useState(true)
	const [counter, setCounter] = React.useState(15)
	const [loadingMessage, setLoadingMessage] = React.useState('Loading')
	const [userEmail,setUserEmail] = React.useState('')

    function redirectToLogin() {
        nav('/login')
    }

    async function fetchDailySchoonover() {
        let response = ''
        if (stateToken !== '') {
            response = await getDailySchoonover(stateToken)
        } else {
            response = await getDailySchoonover(fbToken)
        }
        setMarketData(response)
    }

    function loadContent(){
        if (fbToken !== '') {
            fetchDailySchoonover().then(value => {
                //IsLoading = false
                setLoadingMessage('Updated as of: ' + moment.tz('America/New_York').format('YYYY-MM-DD HH:mm') + ' ')
                setLoading(false)
            })
        }
    }

    useEffect(() => {
        const timer =
            counter > 0 && setInterval(() => setCounter(counter - 1), 1000)
        return () => clearInterval(timer)
    }, [counter])

    useEffect(() => {
        const auth = getAuth()
        onAuthStateChanged(auth, (user) => {
            if (user) {
                user.getIdToken(true).then((token) => {
                    setStateToken(token)
                    fbToken = token
                    setUserEmail(user.email)
                    loadContent()
                    console.log(fbToken)
                })
            } else {
                redirectToLogin()
            }
        })
    }, [])

    return (
        <Container fluid class='flex' className='justify-content-center' style={{ fontSize: FONT_SIZE }}>
            <NavBarTop message={loadingMessage} userEmail={userEmail}/>
            <Table striped bordered hover>
                <tbody>
                    <tr><td>Average Par Traded Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.average_par_traded_today)}</td></tr>
                    <tr><td>Average Price for 3% Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_price_for_3_percent_coupon_bonds}</td></tr>
                    <tr><td>Average Price for 4% Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_price_for_4_percent_coupon_bonds}</td></tr>
                    <tr><td>Average Price for 5% Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_price_for_5_percent_coupon_bonds}</td></tr>
                    <tr><td>Average Price Zero Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_price_for_zero_coupon_bonds}</td></tr>
                    <tr><td>Average Yield for 3% Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_yield_for_3_percent_coupon_bonds}</td></tr>
                    <tr><td>Average Yield for 4% Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_yield_for_4_percent_coupon_bonds}</td></tr>
                    <tr><td>Average Yield for 5% Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_yield_for_5_percent_coupon_bonds}</td></tr>
                    <tr><td>Average Yield for Zero Coupon Bonds:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.average_yield_for_zero_coupon_bonds}</td></tr>
                    <tr><td>Number of Cancelled Trades Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_cancelled_trades_today)}</td></tr>
                    <tr><td>Most Actively Traded New Issue Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.most_actively_traded_new_issue_today.toUpperCase()}</td></tr>
                    <tr><td>Most Actively Traded Seasoned Issue Description:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.most_actively_traded_seasoned_issue_description}</td></tr>
                    <tr><td>Number of Modified Trades Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_modified_trades_today)}</td></tr>
                    <tr><td>Most Actively Traded New Issue Description:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.most_actively_traded_new_issue_description}</td></tr>
                    <tr><td>Most Actively Traded Seasoned Issue Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:daily_schoonover_data.most_actively_traded_seasoned_issue_today.toUpperCase()}</td></tr>
                    <tr><td>Number of Customer Bought Trades Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_customer_buy_trades_today)}</td></tr>
                    <tr><td>Number of customer Bought Trades over $5mm:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_customer_buy_trades_over_5mm_dollars_today)}</td></tr>
                    <tr><td>Number of Customer Sold Trades Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_customer_sell_trades_today)}</td></tr>
                    <tr><td>Number of Customer Sold Trades $5mm:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_customer_sold_trades_over_5mm_dollars_today)}</td></tr>
                    <tr><td>Number of Dealer to Dealer Trades Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_dealer_trades)}</td></tr>
                    <tr><td>Number of Dealer to Dealer trades over $5mm:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_dealer_trades_over_5mm_dollars_today)}</td></tr>
                    <tr><td>Total Number of Trades this Year</td><td>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_trades_this_year)}</td></tr>
                    <tr><td>Total Volume of Trades Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.total_trade_volume_today)}</td></tr>
                    <tr><td>Number of Trades Today:</td><td class='w-50'>{isLoading?<Placeholder animation='glow'><Placeholder xs={7} /></Placeholder>:Intl.NumberFormat('en-US').format(daily_schoonover_data.number_of_trades_today)}</td></tr>
                </tbody>
            </Table>

        </Container>
    )
}


export default (DailySchoonover)