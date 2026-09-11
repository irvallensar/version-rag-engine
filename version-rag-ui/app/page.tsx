'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type DocumentSource = {
  doc_id: number;
  title: string;
  year: string;
  url: string;
  snippet: string;
};

type LatencyBreakdown = {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
};

type Message = {
  role: 'user' | 'assistant';
  content: string;
  latency?: LatencyBreakdown;
  sources?: DocumentSource[];
};

export default function Home() {
  const [query, setQuery] = useState('');
  const [carModel, setCarModel] = useState('735i M Sport');
  const [modelYear, setModelYear] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedSource, setSelectedSource] = useState<DocumentSource | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleClearChat = () => {
    setMessages([]);
    setError('');
    setSelectedSource(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const currentQuery = query;
    setMessages((prev) => [...prev, { role: 'user', content: currentQuery }]);
    setQuery('');
    setIsLoading(true);
    setError('');

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${apiUrl}/api/v1/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentQuery,
          manufacturer: 'BMW',
          model: carModel,
          model_year: modelYear === '' ? null : modelYear,
          top_k: 5,
          history: messages.map((m) => ({ role: m.role, content: m.content })),
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          latency: data.latency,
          sources: data.sources,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center py-6 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="max-w-5xl w-full flex flex-col h-[92vh]">
        
        {/* Header with Clear Chat */}
        <header className="flex justify-between items-center pb-4 border-b border-slate-800 flex-shrink-0">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse"></span>
              BMW Technical Intelligence Engine
            </h1>
            <p className="text-ms text-slate-400 mt-0.5">
              Hybrid Dense/Sparse Retrieval with Cross-Encoder Validation
            </p>
          </div>
          <button
            onClick={handleClearChat}
            className="px-3 py-1.5 text-xs font-semibold rounded border border-slate-800 bg-blue-950 hover:bg-blue-800 text-slate-300 transition"
          >
            Clear Thread
          </button>
        </header>

        {/* Chat Thread */}
        <div className="flex-1 overflow-y-auto py-6 space-y-6 pr-2">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 text-sm">
              <div className="border border-slate-800 rounded-lg p-6 max-w-sm text-center bg-slate-950/40">
                <p className="font-medium text-slate-400 mb-1 text-xl">Get Started</p>
                <p className="text-md text-slate-500">
                  Select vehicle configurations below to benchmark document-grounded specifications.
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-lg px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-950 border border-slate-800 text-slate-200 shadow-sm'
                  }`}
                >
                  {msg.role === 'user' ? (
                    <p className="text-md text-[15px] whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div>
                      <div className="prose prose-invert prose-sm max-w-none text-md text-[15px] text-slate-200">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>

                      {/* Source Badges */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap gap-2 items-center">
                          <span className="text-[12px] uppercase font-semibold text-slate-500 tracking-wider">
                            Sources:
                          </span>
                          {msg.sources.map((src) => (
                            <button
                              key={src.doc_id}
                              onClick={() => setSelectedSource(src)}
                              className="text-[11px] px-2 py-0.5 rounded bg-slate-900 border border-slate-700 hover:border-blue-500 hover:text-blue-400 text-slate-400 transition flex items-center gap-1"
                            >
                              <span>Doc {src.doc_id}</span>
                              <span className="text-[9px] text-slate-500">({src.year})</span>
                            </button>
                          ))}
                        </div>
                      )}

                      {/* Execution Metrics Badges */}
                      {msg.latency && (
                        <div className="mt-2 text-[11px] text-slate-500 font-mono flex gap-3">
                          <span>Retrieval: {msg.latency.retrieval_ms}ms</span>
                          <span>Inference: {msg.latency.generation_ms}ms</span>
                          <span>Total: {msg.latency.total_ms}ms</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex items-start">
              <div className="bg-slate-950 border border-slate-800 text-slate-400 text-sm px-4 py-3 rounded-lg flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping"></span>
                Processing query through hybrid reranking pipeline...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Source Inspector Modal / Drawer */}
        {selectedSource && (
          <div className="mb-4 p-3 bg-slate-950 border border-blue-900/50 rounded-lg text-xs relative animate-in fade-in slide-in-from-bottom-2">
            <button
              onClick={() => setSelectedSource(null)}
              className="absolute top-2 right-2 text-slate-500 hover:text-slate-300 font-bold"
            >
              ✕
            </button>
            <div className="font-semibold text-blue-400 flex items-center gap-2">
              <span>Document {selectedSource.doc_id}: {selectedSource.title} ({selectedSource.year})</span>
              <a
                href={selectedSource.url}
                target="_blank"
                rel="noreferrer"
                className="text-[10px] underline text-slate-500 hover:text-slate-400"
              >
                External PDF Link
              </a>
            </div>
            <p className="mt-1 font-mono text-[11px] text-slate-300 bg-slate-900 p-2 rounded border border-slate-800/80">
              {selectedSource.snippet}
            </p>
          </div>
        )}

        {/* Error Notification */}
        {error && (
          <div className="mb-4 p-2.5 bg-red-950/60 border border-red-800 rounded text-red-300 text-xs flex-shrink-0">
            {error}
          </div>
        )}

        {/* Control Bar */}
        <form onSubmit={handleSubmit} className="flex-shrink-0 pt-4 border-t border-slate-800 py-2">
          <div className="flex flex-col sm:flex-row gap-2 ">
            <select
              value={carModel}
              onChange={(e) => setCarModel(e.target.value)}
              className="h-10 bg-slate-800 border border-slate-700 text-sm rounded-md px-3 text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 hover:bg-slate-700 text-slate-300 transition"
            >
              <option value="735i M Sport">735i M Sport</option>
              <option value="i7 xDrive60">i7 xDrive60</option>
              <option value="730Li M Sport">730Li M Sport</option>
              <option value="740Li Opulence">740Li Opulence</option>
            </select>

            <select
              value={modelYear}
              onChange={(e) => setModelYear(e.target.value)}
              className="h-10 bg-slate-800 border border-slate-700  text-sm rounded-md px-3 py-2 text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 hover:bg-slate-700 text-slate-300 transition"
            >
              <option value="">Auto-Detect Year</option>
              <option value="2025">2025</option>
              <option value="2024">2024</option>
              <option value="2023">2023</option>
              <option value="2022">2022</option>
              <option value="2021">2021</option>
            </select>

            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Query vehicle specifications..."
              className="flex-1 bg-slate-950 border border-slate-800 text-sm rounded-md px-3 py-2 text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 placeholder-slate-600"
              disabled={isLoading}
            />

            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="bg-blue-900 text-white text-sm font-semibold px-4 py-2 rounded-md focus:outline-none hover:bg-blue-700 transition"
            >
              Submit
            </button>
          </div>
        </form>

      </div>
    </main>
  );
}