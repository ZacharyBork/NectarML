import asyncio
import json
import weakref
from pathlib import Path
from aiohttp import web

STATIC_DIR = Path(__file__).parent / 'static'

class VizServer:
    def __init__(
        self, 
        host: str = '0.0.0.0', 
        port: int = 8097
    ) -> None:
        self.host = host
        self.port = port
        self._clients: weakref.WeakSet = weakref.WeakSet()
        self._state: dict = {}

    async def _ws_handler(
        self, 
        request: web.BaseRequest
    ) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)

        for msg in self._state.values(): await ws.send_str(json.dumps(msg))
        async for msg in ws: pass
        return ws

    async def _push_handler(self, request: web.BaseRequest) -> web.Response:
        data = await request.json()
        win = data.get('win', 'default')
        
        if data['type'] == 'clear':
            self._state.clear()
            for ws in self._clients:
                await ws.send_str(json.dumps({'type': 'clear'}))
            return web.json_response({'ok': True})
        
        if data['type'] == 'line_update' and win in self._state:
            existing = self._state[win]
            existing['X'].extend(data['X'])
            for i, series in enumerate(data['Y']):
                existing['Y'][i].extend(series)
            broadcast = existing
        else: broadcast = self._state[win] = data

        dead = set()
        for ws in self._clients:
            try: await ws.send_str(json.dumps(broadcast))
            except Exception: dead.add(ws)
        self._clients -= dead

        return web.json_response({'ok': True})

    async def _favicon_handler(self, request: web.BaseRequest) -> None:
        raise web.HTTPNoContent()

    async def _index_handler(
        self, 
        request: web.BaseRequest
    ) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / 'index.html')

    def run(self) -> None:
        app = web.Application(client_max_size=50 * 1024 * 1024)
        app.router.add_get('/favicon.ico', self._favicon_handler)
        app.router.add_get('/', self._index_handler)
        app.router.add_get('/ws', self._ws_handler)
        app.router.add_post('/push', self._push_handler)
        app.router.add_static('/', STATIC_DIR)
        web.run_app(app, host=self.host, port=self.port)

if __name__ == '__main__':
    server = VizServer()
    server.run()
