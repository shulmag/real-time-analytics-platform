// change this to get, not post
import http from './httpService'

export async function getYield(access_token,date,time) {
    const urlStr = '/api/yield?access_token=' + access_token + '&date=' + date + '&time=' + time 
    const { data: returnedData} = await http.get(urlStr)
    //debugger
    return returnedData
}

  