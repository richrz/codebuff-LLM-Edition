
export type ChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  name?: string
}

export type ChatStreamChunk =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; text: string }

export type ChatOptions = {
  temperature?: number
  stream?: boolean
  maxTokens?: number
  stop?: string | string[]
  topP?: number
  frequencyPenalty?: number
  presencePenalty?: number
  signal?: AbortSignal
}

function requireEnv(name: 'LLM_BASE_URL' | 'LLM_API_KEY' | 'LLM_MODEL'): string {
  const value = process.env[name]
  if (!value) {
    throw new Error(`${name} environment variable is not set`)
  }
  return value
}

function buildPayload(
  messages: ChatMessage[],
  options: ChatOptions,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    model: requireEnv('LLM_MODEL'),
    messages,
  }

  if (options.temperature !== undefined) {
    payload.temperature = options.temperature
  }
  if (options.maxTokens !== undefined) {
    payload.max_tokens = options.maxTokens
  }
  if (options.stop !== undefined) {
    payload.stop = options.stop
  }
  if (options.topP !== undefined) {
    payload.top_p = options.topP
  }
  if (options.frequencyPenalty !== undefined) {
    payload.frequency_penalty = options.frequencyPenalty
  }
  if (options.presencePenalty !== undefined) {
    payload.presence_penalty = options.presencePenalty
  }
  if (options.stream) {
    payload.stream = true
  }

  return payload
}

async function request(
  messages: ChatMessage[],
  options: ChatOptions,
) {
  const baseUrl = requireEnv('LLM_BASE_URL').replace(/\/$/, '')
  const apiKey = requireEnv('LLM_API_KEY')
  const url = `${baseUrl}/v1/chat/completions`

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(buildPayload(messages, options)),
    signal: options.signal,
  })

  if (!response.ok) {
    const errorBody = await response.text().catch(() => '')
    const message = errorBody
      ? `LLM request failed (${response.status}): ${errorBody}`
      : `LLM request failed with status ${response.status}`
    throw new Error(message)
  }

  return response
}

async function* parseStream(
  response: Response,
): AsyncGenerator<ChatStreamChunk, void, unknown> {
  const body = response.body
  if (!body) {
    throw new Error('Streaming response did not include a readable body')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  for await (const chunk of body as AsyncIterable<Uint8Array>) {
    buffer += decoder.decode(chunk, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)

      const dataLines = rawEvent
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .filter((line) => line.length > 0)

      for (const data of dataLines) {
        if (data === '[DONE]') {
          return
        }

        let parsed: any
        try {
          parsed = JSON.parse(data)
        } catch {
          continue
        }

        const delta = parsed?.choices?.[0]?.delta
        if (!delta) {
          continue
        }

        const reasoning = delta.reasoning
        if (Array.isArray(reasoning)) {
          for (const item of reasoning) {
            if (item?.type === 'reasoning' && typeof item?.text === 'string') {
              yield { type: 'reasoning', text: item.text }
            }
          }
        } else if (
          reasoning &&
          typeof reasoning === 'object' &&
          typeof reasoning.text === 'string'
        ) {
          yield { type: 'reasoning', text: reasoning.text }
        }

        const content = delta.content
        if (typeof content === 'string') {
          if (content.length > 0) {
            yield { type: 'text', text: content }
          }
        } else if (Array.isArray(content)) {
          for (const part of content) {
            if (part?.type === 'text' && typeof part?.text === 'string') {
              yield { type: 'text', text: part.text }
            } else if (
              part?.type === 'reasoning' &&
              typeof part?.text === 'string'
            ) {
              yield { type: 'reasoning', text: part.text }
            } else if (typeof part === 'string' && part.length > 0) {
              yield { type: 'text', text: part }
            }
          }
        }
      }

      boundary = buffer.indexOf('\n\n')
    }
  }

  buffer += decoder.decode()
  let boundary = buffer.indexOf('\n\n')
  while (boundary !== -1) {
    const rawEvent = buffer.slice(0, boundary)
    buffer = buffer.slice(boundary + 2)

    const dataLines = rawEvent
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
      .filter((line) => line.length > 0)

    for (const data of dataLines) {
      if (data === '[DONE]') {
        return
      }

      let parsed: any
      try {
        parsed = JSON.parse(data)
      } catch {
        continue
      }

      const delta = parsed?.choices?.[0]?.delta
      if (!delta) {
        continue
      }

      const reasoning = delta.reasoning
      if (Array.isArray(reasoning)) {
        for (const item of reasoning) {
          if (item?.type === 'reasoning' && typeof item?.text === 'string') {
            yield { type: 'reasoning', text: item.text }
          }
        }
      } else if (
        reasoning &&
        typeof reasoning === 'object' &&
        typeof reasoning.text === 'string'
      ) {
        yield { type: 'reasoning', text: reasoning.text }
      }

      const content = delta.content
      if (typeof content === 'string') {
        if (content.length > 0) {
          yield { type: 'text', text: content }
        }
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (part?.type === 'text' && typeof part?.text === 'string') {
            yield { type: 'text', text: part.text }
          } else if (
            part?.type === 'reasoning' &&
            typeof part?.text === 'string'
          ) {
            yield { type: 'reasoning', text: part.text }
          } else if (typeof part === 'string' && part.length > 0) {
            yield { type: 'text', text: part }
          }
        }
      }
    }

    boundary = buffer.indexOf('\n\n')
  }
}

async function parseResponse(response: Response): Promise<string> {
  const payload = (await response.json()) as any
  const choice = payload?.choices?.[0]
  const message = choice?.message
  const content = message?.content

  if (!content) {
    return ''
  }

  if (typeof content === 'string') {
    return content
  }

  if (Array.isArray(content)) {
    return content
      .map((part: any) => {
        if (part?.type === 'text' && typeof part?.text === 'string') {
          return part.text
        }
        if (typeof part === 'string') {
          return part
        }
        return ''
      })
      .filter((text: string) => text.length > 0)
      .join('')
  }

  return ''
}

export async function chat(
  messages: ChatMessage[],
  options: ChatOptions = {},
): Promise<string>
export async function chat(
  messages: ChatMessage[],
  options: ChatOptions & { stream: true },
): Promise<AsyncGenerator<ChatStreamChunk, void, unknown>>
export async function chat(
  messages: ChatMessage[],
  options: ChatOptions = {},
): Promise<string | AsyncGenerator<ChatStreamChunk, void, unknown>> {
  const response = await request(messages, options)

  if (options.stream) {
    return parseStream(response)
  }

  return parseResponse(response)
}
