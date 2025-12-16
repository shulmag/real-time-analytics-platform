import http from './httpService'

export async function getCusipTable(access_token, coupon, desc, from, to) {
    const urlStr = '/api/cusiptable?access_token=' + access_token + '&coupon=' + coupon + '&desc=' + desc + '&from=' + from + '&to=' + to 
    const { data: returnedData } = await http.get(urlStr)
    return returnedData
}