import http from './httpService'

export async function getYTWCurve(access_token, cusip, tradeType, amount, date, time) {
    const urlStr = '/api/ytwCurve?access_token=' + access_token + '&cusip=' + cusip + '&tradeType='+tradeType + '&amount=' + amount + '&date=' + date + '&time=' + time 
    const { data: returnedData } = await http.get(urlStr)
    return returnedData
}
