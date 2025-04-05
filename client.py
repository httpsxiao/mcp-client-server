import asyncio
import os
import sys
import json
import re
from typing import Optional
from contextlib import AsyncExitStack

# LLM 接入
from openai import OpenAI
from dotenv import load_dotenv

# mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 加载 .env 文件，确保 API Key 受到保护
load_dotenv()


class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")  # 读取 OpenAI API Key
        self.base_url = os.getenv("BASE_URL")  # 读取 BASE YRL
        self.model = os.getenv("MODEL")  # 读取 model
        if not self.openai_api_key:
            raise ValueError(
                "❌ 未找到 OpenAI API Key，请在 .env 文件中设置OPENAI_API_KEY"
            )
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务器并列出可用工具"""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是.py或.js文件")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )
        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器,支持以下工具:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """使用大模型处理查询并调用可用的 MCP 工具"""
        messages = [
            {"role": "user", "content": query},
        ]
        response = await self.session.list_tools()

        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]

        print("\n available_tools 可用工具:", available_tools)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools,
        )

        # 处理返回的内容
        modelResult = response.choices[0]

        if modelResult.finish_reason == "tool_calls":
            # 如何是需要使用工具，就解析工具
            tool_call = modelResult.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 执行工具
            toolResult = await self.session.call_tool(tool_name, tool_args)
            print(
                f"\n\n[Callingtool {tool_name} with args {tool_args}]\n toolResult is {toolResult.content[0].text}\n"
            )

            # 将模型返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
            messages.append(modelResult.message.model_dump())
            messages.append(
                {
                    "role": "tool",
                    "content": toolResult.content[0].text,
                    "tool_call_id": tool_call.id,
                }
            )

            # 将上面的结果再返回给大模型用于生产最终的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            rmThinkContent = re.sub(
                r"<think>.*?</think>",
                "",
                response.choices[0].message.content,
                flags=re.DOTALL,
            )
            return rmThinkContent

        return modelResult.message.content

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动！输入 'quit' 退出")
        while True:
            try:
                query = input("\n请输入内容:").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)  # 发送用户输入到大模型
                print(f"\n🤖 大模型: {response}")
            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("python client.py<path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
