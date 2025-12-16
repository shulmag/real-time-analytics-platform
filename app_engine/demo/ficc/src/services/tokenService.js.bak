import http from './httpService'
// change this to get, not post

export async function getToken(userIdToken) {
    const urlStr = '/api/login?access_token=' + userIdToken
    const { data: returnedData} = await http.get(urlStr)
    /*const { data: returnedData} = await http.get(urlStr,{
      headers: {
        'Authorization': 'Bearer ' + userIdToken
      }
    }) */
    return returnedData
}