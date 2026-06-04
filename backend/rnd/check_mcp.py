import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

async def get_remote_server_info(server_url: str):
    """
    Connect to a remote MCP server via SSE and get its information.
    
    Args:
        server_url: The URL of the remote MCP server (e.g., "http://localhost:8000/sse")
    """
    print(f"Connecting to remote MCP server at: {server_url}\n")
    
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            print("=== Remote MCP Server Information ===\n")
            
            # Get server capabilities first
            print(f"Server Version: {session.server_version or 'Not specified'}")
            print(f"Protocol Version: {session.negotiated_version}\n")
            
            # List available tools
            tools_result = await session.list_tools()
            print(f"🔧 Available Tools: {len(tools_result.tools)}\n")
            
            for tool in tools_result.tools:
                print(f"  📌 {tool.name}")
                if tool.description:
                    print(f"     Description: {tool.description[:150]}...")
                print(f"     Input Schema: {json.dumps(tool.inputSchema, indent=2)}")
                print()
            
            # List available prompts
            try:
                prompts_result = await session.list_prompts()
                if prompts_result.prompts:
                    print(f"\n💬 Available Prompts: {len(prompts_result.prompts)}")
                    for prompt in prompts_result.prompts:
                        print(f"     • {prompt.name}: {prompt.description or 'No description'}")
                else:
                    print("\n💬 No prompts available")
            except Exception as e:
                print(f"\n💬 Prompts not supported: {e}")
            
            # List available resources
            try:
                resources_result = await session.list_resources()
                if resources_result.resources:
                    print(f"\n📄 Available Resources: {len(resources_result.resources)}")
                    for resource in resources_result.resources:
                        print(f"     • {resource.uri}: {resource.description or 'No description'}")
                else:
                    print("\n📄 No resources available")
            except Exception as e:
                print(f"\n📄 Resources not supported: {e}")

async def quick_test(server_url: str):
    """Quick test to check if remote server is responsive"""
    try:
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Just get server info
                result = await session.list_tools()
                print(f"✅ Server is online!")
                print(f"   Found {len(result.tools)} tools")
                print(f"\nTool names: {', '.join([t.name for t in result.tools])}")
                return True
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

if __name__ == "__main__":
    # Replace with your remote MCP server URL
    REMOTE_SERVER_URL = "http://172.30./api/mcp"
    
    # Option 1: Just test connectivity
    print("Testing connection...")
    asyncio.run(quick_test(REMOTE_SERVER_URL))
    
    print("\n" + "="*60 + "\n")
    
    # Option 2: Get full server information
    asyncio.run(get_remote_server_info(REMOTE_SERVER_URL))

