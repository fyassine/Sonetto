"use client";

import { useEffect, useState, useRef } from "react";
import { useParams } from "next/navigation";

export default function CustomerPage() {
  const [message, setMessage] = useState<any>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const params = useParams();
  const customerId = params.customer_id as string;

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Create WebSocket connection inside useEffect to prevent reconnection on every render
    if (!ws.current) {
      ws.current = new WebSocket("ws://localhost:8089/fws");
    }

    ws.current.onopen = () => {
      console.log("Connected to the server");
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      const memory = JSON.parse(JSON.stringify(event.data));
      setError(memory.memories);
      setMessage(memory.memories);
    };

    ws.current.onclose = () => {
      console.log("Disconnected from the server");
      setConnected(false);
    };

    return () => {
      ws.current?.close();
      ws.current = null;
    };
  }, [message]); // Remove message from dependency array to prevent infinite re-renders

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-indigo-50 to-blue-100 dark:from-gray-900 dark:to-indigo-950">
      <div className="w-full max-w-md">
        {/* Connection status indicator */}
        <div className="mb-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">Customer: {customerId}</h1>
          <div className="flex items-center">
            <div
              className={`h-3 w-3 rounded-full mr-2 ${
                connected ? "bg-green-500" : "bg-red-500"
              }`}
            ></div>
            <span className="text-sm">
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-100 dark:bg-red-900/30 rounded-lg text-red-700 dark:text-red-300 backdrop-blur-sm">
            {error}
          </div>
        )}

        {/* Glassy card UI */}
        <div className="relative overflow-hidden rounded-2xl backdrop-blur-md bg-white/30 dark:bg-gray-800/30 border border-white/50 dark:border-gray-700/50 shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.01]">
          {/* Card content */}
          <div className="p-6">
            {/* Message display */}
            <div className="mb-4">
              <h2 className="text-xl font-semibold mb-2">
                {message && message.memory
                  ? "Learned something new"
                  : "Learned nothing new so far"}
              </h2>
              <p className="text-gray-700 dark:text-gray-300">
                {message && message.memory
                  ? message.memory
                  : "No memory available yet"}
              </p>
            </div>
          </div>

          {/* Decorative elements for the glassy effect */}
          <div className="absolute -top-10 -right-10 h-40 w-40 rounded-full bg-blue-300/20 dark:bg-blue-500/10 blur-xl"></div>
          <div className="absolute -bottom-8 -left-8 h-32 w-32 rounded-full bg-purple-300/20 dark:bg-purple-500/10 blur-xl"></div>
        </div>
      </div>
    </div>
  );
}
