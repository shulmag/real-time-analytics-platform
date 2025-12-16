// just axios
// set your base url through env.devlopment or env.production
import axios from 'axios'

// Decide if you are testing App Engine or your local Dev server: 

// axios.defaults.baseURL = 'http://localhost:5000'
// axios.defaults.baseURL = 'https://server-dot-eng-reactor-287421.uc.r.appspot.com'
// axios.defaults.baseURL = 'https://server-3ukzrmokpq-uc.a.run.app'
axios.defaults.baseURL = 'https://api.ficc.ai'

axios.interceptors.response.use(null, error => {
    const expectedError =
        error.response &&
        error.response.status >= 400 &&
        error.response.status < 500

    return Promise.reject(expectedError)
})

export default {
    get: axios.get,
    post: axios.post,
    put: axios.put,
    delete: axios.delete
}
