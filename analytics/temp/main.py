def app(environ, start_response):
    start_response('404 Not Found', [('Content-Type', 'text/plain')])
    return [b'Service disabled']
