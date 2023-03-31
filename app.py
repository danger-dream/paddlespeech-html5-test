import json

import uvicorn
from fastapi import FastAPI
from fastapi import APIRouter
from fastapi import WebSocket
from starlette.websockets import WebSocketState as WebSocketState

from paddlespeech.server.engine.text.python.text_engine import PaddleTextConnectionHandler
from starlette.middleware.cors import CORSMiddleware
from paddlespeech.server.engine.asr.online.python.asr_engine import ASREngine
from paddlespeech.server.engine.text.python.text_engine import TextEngine

port = 80

app = FastAPI(title="Serving API", description="Api", version="0.0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

router = APIRouter()

class D2o(dict):
	def __getattr__(self, name):
		value = self[name]
		if isinstance(value, dict):
			return D2o(value)
		return value

asr_engine = ASREngine()
text_engine = TextEngine()

# asr引擎参数
asr_engine.init(config=D2o({
	"model_type": "conformer_online_multicn",
	"am_model": None,
	"am_params": None,
	"lang": "zh",
	"sample_rate": 16000,
	"cfg_path": None,
	"decode_method": None,
	"num_decoding_left_chunks": -1,
	"force_yes": True,
	"device": "cpu",
	"continuous_decoding": True,
	"am_predictor_conf": {
		"device": "cpu",
		"switch_ir_optim": True,
		"glog_info": False,
		"summary": True
	},
	"chunk_buffer_conf": {
		"window_n": 7,
		"shift_n": 4,
		"window_ms": 25,
		"shift_ms": 10,
		"sample_rate": 16000,
		"sample_width": 2
	}
}))

# 标点符号恢复参数
text_engine.init(config=D2o({
	"task": "punc",
	"model_type": "ernie_linear_p3_wudao",
	"lang": "zh",
	"sample_rate": 16000,
	"cfg_path": None,
	"ckpt_path": None,
	"vocab_file": None,
	"device": "cpu"
}))

@router.websocket('/asr/streaming')
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()
	
  text_connection_handler = None
	connection_handler = None
		
	try:
		while True:
			
			assert websocket.application_state == WebSocketState.CONNECTED
			message = await websocket.receive()
			websocket._raise_on_disconnect(message)
			if "text" in message:
				message = json.loads(message["text"])
				if 'signal' not in message:
					await websocket.send_json({ "status": "ok", "message": "no valid json data" })
				if message['signal'] == 'start':
					# 开始识别，初始化引擎
					connection_handler = asr_engine.new_handler()
					text_connection_handler = PaddleTextConnectionHandler(text_engine)
					await websocket.send_json({ "status": "ok", "signal": "server_ready" })
				elif message['signal'] == 'end':
					connection_handler.decode(is_finished=True)
					connection_handler.rescoring()
					asr_results = connection_handler.get_result()
					word_time_stamp = connection_handler.get_word_time_stamp()
					connection_handler.reset()
					await websocket.send_json({ "status": "ok", "signal": "finished", 'result': asr_results, 'times': word_time_stamp })
					break
				else:
					resp = { "status": "ok", "message": "no valid json data" }
					await websocket.send_json(resp)
			
			elif "bytes" in message:
				message = message["bytes"]
				# 提取特征
				connection_handler.extract_feat(message)
				# 解码
				connection_handler.decode(is_finished=False)
				
				if connection_handler.endpoint_state:
					connection_handler.rescoring()
					word_time_stamp = connection_handler.get_word_time_stamp()
				# 取预测结果
				asr_results = connection_handler.get_result()
				if asr_results:
					try:
						# 恢复标点符号
						punc_text = text_connection_handler.run(asr_results)
						if punc_text is not None:
							asr_results = punc_text
					except BaseException:
						pass
				
				if connection_handler.endpoint_state:
					if connection_handler.continuous_decoding:
						connection_handler.reset_continuous_decoding()
					else:
						await websocket.send_json({ "status": "ok", "signal": "finished", "result": asr_results, "times": word_time_stamp })
						break
				resp = { 'result': asr_results }
				await websocket.send_json(resp)
			else:
				print("Received data of unrecognized type")

	except BaseException as e:
		print("error： ", e)

if __name__ == "__main__":
	app.include_router(router)
	uvicorn.run(app, host='0.0.0.0', port=port)
