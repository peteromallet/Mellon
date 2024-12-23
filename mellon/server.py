from aiohttp import web, WSMsgType
from aiohttp_cors import setup as cors_setup, ResourceOptions
import json
import nanoid
import io
import base64
import re
from importlib import import_module
import logging
logger = logging.getLogger('mellon')
import asyncio
import traceback
from utils.memory_manager import memory_flush
from copy import deepcopy

class WebServer:
    def __init__(self, module_map: dict, host: str = "0.0.0.0", port: int = 8080, cors: bool = False, cors_route: str = "*"):
        self.module_map = module_map
        self.node_store = {}
        self.queue = asyncio.Queue()
        self.queue_task = None
        self.host = host
        self.port = port
        self.ws_clients = {}
        self.app = web.Application()
        self.event_loop = None

        self.progress_queue = asyncio.Queue()
        self.progress_task = None

        self.app.add_routes([web.get('/', self.index),
                             web.get('/nodes', self.nodes),
                             web.get('/custom_component/{module}/{component}', self.custom_component),
                             web.get('/custom_assets/{module}/{file_path}', self.custom_assets),
                             web.post('/graph', self.graph),
                             web.delete('/clearNodeCache', self.clear_node_cache),
                             web.static('/assets', 'web/assets'),
                             web.get('/favicon.ico', self.favicon),
                             web.get('/ws', self.websocket_handler)])

        if cors:
            cors = cors_setup(self.app, defaults={
                cors_route: ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            })

            for route in list(self.app.router.routes()):
                cors.add(route)

    def run(self):
        # TODO: need to do proper queue processing
        async def start_app():
            self.event_loop = asyncio.get_event_loop()
            runner = web.AppRunner(self.app, client_max_size=1024**4)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            
            # Start queue processor
            self.queue_task = asyncio.create_task(self.process_queue())
            self.progress_task = asyncio.create_task(self.process_progress())
            
            await site.start()
            
            try:
                await asyncio.Future()
            finally:
                # Cleanup
                if self.queue_task:
                    self.queue_task.cancel()
                if self.progress_task:
                    self.progress_task.cancel()
                await runner.cleanup()
        
        asyncio.run(start_app())

    async def process_progress(self):
        while True:
            item = await self.progress_queue.get()
            try:
                await self.ws_clients[item["client_id"]].send_json({
                    "type": "progress",
                    "nodeId": item["nodeId"],
                    "progress": item["progress"]
                })
                #await asyncio.sleep(0.02)
            except Exception as e:
                logger.error(f"Error sending progress update: {str(e)}")
            finally:
                self.progress_queue.task_done()

    async def process_queue(self):
        while True:
            item = await self.queue.get()
            try:
                await self.graph_execution(item)
            except Exception as e:
                logger.error(f"Error processing queue task: {str(e)}")
                logger.error(f"Error occurred in {traceback.format_exc()}")
                await self.broadcast({
                    "type": "error",
                    "error": "An unexpected error occurred while processing the graph"
                })
            finally:
                self.queue.task_done()

    async def index(self, request):
        response = web.FileResponse('web/index.html')
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def favicon(self, request):
        return web.FileResponse('web/favicon.ico')

    """
    HTTP API
    """

    async def custom_component(self, request):
        module = request.match_info.get('module')
        component = request.match_info.get('component')

        path = component.split('/')
        if len(path) > 1:
            module = path[0]
            component = path[1]

        if module not in self.module_map:
            raise web.HTTPNotFound(text=f"Module {module} not found")
        
        response = web.FileResponse(f'modules/{module}/web/{component}.js')
        response.headers["Content-Type"] = "application/javascript"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def custom_assets(self, request):
        module = request.match_info.get('module')
        file_path = request.match_info.get('file_path')

        if module not in self.module_map:
            raise web.HTTPNotFound(text=f"Module {module} not found")

        return web.FileResponse(f'modules/{module}/web/assets/{file_path}')

    async def nodes(self, request):
        nodes = {}
        for module_name, actions in self.module_map.items():
            for action_name, action in actions.items():
                params = {}
                if 'params' in action:
                    params = deepcopy(action['params'])
                    # remove attributes that are not needed by the client
                    for p in params:
                        if 'postProcess' in params[p]:
                            del params[p]['postProcess']

                nodes[f"{module_name}-{action_name}"] = {
                    "label": action['label'] if 'label' in action else f"{module_name}: {action_name}",
                    "module": module_name,
                    "action": action_name,
                    "category": self.slugify(action['category']) if 'category' in action else "default",
                    "params": params,
                }
                if 'style' in action:
                    nodes[f"{module_name}-{action_name}"]["style"] = action['style']

        return web.json_response(nodes)
    
    async def clear_node_cache(self, request):
        data = await request.json()
        nodeId = []

        if "nodeId" in data:
            nodeId = data["nodeId"] if isinstance(data["nodeId"], list) else [data["nodeId"]]
        else:
            nodeId = list(self.node_store.keys())

        for node in nodeId:
            if node in self.node_store:
                del self.node_store[node]

        memory_flush(gc_collect=True)

        return web.json_response({
            "type": "cacheCleared",
            "nodeId": nodeId
        })

    async def graph(self, request):
        graph = await request.json()
        await self.queue.put(graph)
        return web.json_response({
            "type": "graphQueued",
            "sid": graph["sid"]
        })

    async def graph_execution(self, graph):
        #graph = await request.json()
        sid = graph["sid"]
        nodes = graph["nodes"]
        paths = graph["paths"]

        for path in paths:
            for node in path:
                module_name = nodes[node]["module"]
                action_name = nodes[node]["action"]
                logger.debug(f"Executing node {module_name}.{action_name}")

                params = nodes[node]["params"]
                ui_fields = {}
                args = {}
                for p in params:
                    source_id = params[p]["sourceId"] if "sourceId" in params[p] else None
                    source_key = params[p]["sourceKey"] if "sourceKey" in params[p] else None
                    # if there is a source key, it means that the value comes from a pipeline,
                    # so we follow the connection to the source node and get the associated value
                    # Otherwise we use the value in the params
                    
                    if "display" in params[p] and params[p]["display"] == "ui":
                        # store ui fields that need to be sent back to the client
                        if params[p]["type"] == "image":
                            ui_fields[p] = { "source": source_key, "type": params[p]["type"] }
                        elif params[p]["type"] == "3d":
                            ui_fields[p] = { "source": source_key, "type": params[p]["type"] }
                    else:
                        args[p] = self.node_store[source_id].output[source_key] if source_id else params[p]["value"] if 'value' in params[p] else None                        

                if module_name not in self.module_map:
                    raise ValueError("Invalid module")
                if action_name not in self.module_map[module_name]:
                    raise ValueError("Invalid action")
                
                # import the module and get the action
                if module_name.endswith(".custom"):
                    module = import_module(f"custom.{module_name.replace('.custom', '')}.{module_name.replace('.custom', '')}")
                else:
                    module = import_module(f"modules.{module_name}.{module_name}")
                action = getattr(module, action_name)

                # if the node is not in the node store, initialize it
                if node not in self.node_store:
                    self.node_store[node] = action(node)
                
                self.node_store[node]._client_id = sid

                if not callable(self.node_store[node]):
                    raise TypeError(f"The class `{module_name}.{action_name}` is not callable. Ensure that the class has a __call__ method or extend it from `NodeBase`.")
                
                # initialize the progress bar
                await self.ws_clients[sid].send_json({
                    "type": "progress",
                    "nodeId": node,
                    "progress": -1
                })

                try:
                    await self.event_loop.run_in_executor(None, lambda: self.node_store[node](**args))
                    #self.node_store[node](**args)
                except Exception as e:
                    logger.error(f"Error executing node {module_name}.{action_name}: {str(e)}")
                    raise e
                
                execution_time = self.node_store[node]._execution_time if hasattr(self.node_store[node], '_execution_time') else 0

                updateValues = {}
                if "onAfterNodeExecute" in self.module_map[module_name][action_name]:
                    for event in self.module_map[module_name][action_name]["onAfterNodeExecute"]:
                        if event["action"] == "updateValue":
                            updateValues.update({ event["target"]: self.node_store[node].params[event["target"]] })

                await self.ws_clients[sid].send_json({
                    "type": "executed",
                    "nodeId": node,
                    "time": f"{execution_time:.2f}",
                    "updateValues": updateValues
                    #"memory": f"{memory_usage/1024**3:.2f}"
                })

                logger.debug(f"Node {module_name}.{action_name} executed in {execution_time:.3f}s")

                # TODO: this is just a placeholder for now

                for key in ui_fields:
                    value = self.node_store[node].output[ui_fields[key]["source"]]
                    await self.ws_clients[sid].send_json({
                        "type": ui_fields[key]["type"],
                        "key": key,
                        "nodeId": node,
                        "data": self.to_base64(ui_fields[key]["type"], value)
                    })


    """
    WebSocket API
    """

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.query.get("sid")
        logger.info(f"WebSocket connection with sid {sid}")
        if sid:
            if sid in self.ws_clients:
                del self.ws_clients[sid]
        else:
            # if the client does not provide a session id, we create one for them one
            sid = nanoid.generate(size=10)

        self.ws_clients[sid] = ws
        await ws.send_json({"type": "welcome", "sid": sid})

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)

                try:
                    if data["type"] == "ping":
                        await ws.send_json({"type": "pong"})
                    elif data["type"] == "close":
                        await ws.close()
                        break
                        """
                    elif data["type"] == "module":
                        module_name = data["module"]
                        action_name = data["action"]
                        params = data["data"] if "data" in data else {}
                        
                        if module_name not in self.module_map or action_name not in self.module_map[module_name]:
                            raise ValueError("Invalid module or action")
                        
                        module = import_module(f"modules.{module_name}.{module_name}")
                        action = getattr(module, action_name)
                        result = await action(**params)
                        await ws.send_json({"type": "result", "result": result})
                    
                    elif data["type"] == "graph":
                        graph = data["graph"]
                        for node in graph["nodes"]:
                            module_name = node["module"]
                            action_name = node["action"]
                            params = node["params"]
                            module = import_module(f"modules.{module_name}.{module_name}")
                            action = getattr(module, action_name)
                            result = await action(**params)
                            await ws.send_json({"type": "result", "result": result})
                        """
                    else:
                        raise ValueError("Invalid message type")

                #except KeyError as e:
                #    await ws.send_json({"type": "error", "message": f"Missing required field: {str(e)}"})
                #except ValueError as e:
                #    await ws.send_json({"type": "error", "message": str(e)})
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    await ws.send_json({"type": "error", "message": "An unexpected error occurred"})
            
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket connection closed with exception {ws.exception()}')

        del self.ws_clients[sid]
        logger.info(f'WebSocket connection {sid} closed')
            
        return ws
    
    async def broadcast(self, message, client_id=None):
        if client_id:
            if client_id not in self.ws_clients:
                return
            ws_clients = [client_id] if not isinstance(client_id, list) else client_id
        else:
            ws_clients = self.ws_clients

        for client in ws_clients:
            await self.ws_clients[client].send_json(message)

    
    """
    Helper functions
    """
    def to_base64(self, type, value):
        if type == "image":
            img_byte_arr = io.BytesIO()
            value.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            return base64.b64encode(img_byte_arr).decode('utf-8')
        elif type == "3d":
            glb_byte_arr = io.BytesIO()
            glb_byte_arr.write(value)
            glb_byte_arr = glb_byte_arr.getvalue()
            return base64.b64encode(glb_byte_arr).decode('utf-8')
    
    def slugify(self, text):
        return re.sub(r'[^\w\s-]', '', text).strip().replace(' ', '-')

from modules import MODULE_MAP
from config import config

web_server = WebServer(MODULE_MAP, **config.server)
