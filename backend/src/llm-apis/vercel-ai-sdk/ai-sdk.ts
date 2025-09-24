import { buildArray } from '@codebuff/common/util/array'
import { convertCbToModelMessages } from '@codebuff/common/util/messages'
import { errorToObject } from '@codebuff/common/util/object'
import { withTimeout } from '@codebuff/common/util/promise'
import { StopSequenceHandler } from '@codebuff/common/util/stop-sequence'
import { generateCompactId } from '@codebuff/common/util/string'
import { encode } from 'gpt-tokenizer'

import { checkLiveUserInput, getLiveUserInputIds } from '../../live-user-inputs'
import { logger } from '../../util/logger'
import { saveMessage } from '../message-cost-tracker'
import { chat } from '../../llm/openaiCompatible'

import type {
  ChatMessage,
  ChatStreamChunk,
} from '../../llm/openaiCompatible'
import type { Model } from '@codebuff/common/old-constants'
import type { Message } from '@codebuff/common/types/messages/codebuff-message'
import type { z } from 'zod/v4'

export type StreamChunk =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'error'; message: string }

type ProviderOptions = {
  openrouter?: { reasoning?: { exclude?: boolean } } & Record<string, unknown>
} & Record<string, unknown>

type SharedOptions = {
  messages: Message[]
  clientSessionId: string
  fingerprintId: string
  userInputId: string
  model: Model
  userId: string | undefined
  chargeUser?: boolean
  agentId?: string
  onCostCalculated?: (credits: number) => Promise<void>
  includeCacheControl?: boolean
  temperature?: number
  topP?: number
  frequencyPenalty?: number
  presencePenalty?: number
}

type PromptAiSdkStreamOptions = SharedOptions & {
  stopSequences?: string[]
  maxOutputTokens?: number
  providerOptions?: ProviderOptions
  thinkingBudget?: number
  maxRetries?: number
  signal?: AbortSignal
}

type PromptAiSdkOptions = SharedOptions & {
  maxTokens?: number
  stopSequences?: string[]
}

type PromptAiSdkStructuredOptions<T> = SharedOptions & {
  schema: z.ZodType<T>
  maxTokens?: number
  timeout?: number
  stopSequences?: string[]
}

type ConvertedMessage = ReturnType<typeof convertCbToModelMessages>[number]
type ConvertedContent = ConvertedMessage['content']

type ConvertedContentPart = ConvertedContent extends string
  ? never
  : ConvertedContent extends Array<infer Part>
    ? Part
    : never

function toBase64(data: unknown): string | null {
  if (typeof data === 'string') {
    return data
  }
  if (typeof Buffer !== 'undefined' && Buffer.isBuffer(data)) {
    return (data as Buffer).toString('base64')
  }
  if (data instanceof Uint8Array) {
    return Buffer.from(data).toString('base64')
  }
  if (data instanceof ArrayBuffer) {
    return Buffer.from(new Uint8Array(data)).toString('base64')
  }
  return null
}

function stringifyContentPart(part: ConvertedContentPart): string {
  if (!part) {
    return ''
  }

  if (typeof part === 'string') {
    return part
  }

  const anyPart = part as Record<string, unknown>

  if (typeof anyPart.text === 'string') {
    return anyPart.text
  }

  const type = typeof anyPart.type === 'string' ? (anyPart.type as string) : undefined
  if (type === 'file') {
    const base64 = toBase64(anyPart.data)
    if (base64) {
      return base64
    }
    if (typeof anyPart.mediaType === 'string') {
      return `[file:${anyPart.mediaType}]`
    }
    return '[file]'
  }
  if (type === 'image') {
    if (typeof anyPart.image === 'string') {
      return anyPart.image
    }
    if (anyPart.image instanceof URL) {
      return anyPart.image.toString()
    }
    const base64 = toBase64(anyPart.image)
    if (base64) {
      return base64
    }
    if (typeof anyPart.mediaType === 'string') {
      return `[image:${anyPart.mediaType}]`
    }
    return '[image]'
  }
  if (type === 'tool-call') {
    const { toolCallId, toolName, input } = anyPart
    return JSON.stringify({ toolCallId, toolName, input })
  }
  if (typeof anyPart.reasoning === 'string') {
    return anyPart.reasoning
  }

  return ''
}

function stringifyContent(content: ConvertedContent): string {
  if (typeof content === 'string') {
    return content
  }
  return content
    .map((part) => stringifyContentPart(part as ConvertedContentPart))
    .filter((text) => text.length > 0)
    .join('\n')
}

function convertMessagesToChatMessages(
  messages: Message[],
  includeCacheControl?: boolean,
): ChatMessage[] {
  const converted = convertCbToModelMessages({
    messages,
    includeCacheControl: includeCacheControl ?? true,
  })

  return converted.map((message) => {
    const chatMessage: ChatMessage = {
      role: message.role as ChatMessage['role'],
      content: stringifyContent(message.content),
    }

    if ('name' in message && typeof (message as any).name === 'string') {
      chatMessage.name = (message as any).name
    }

    return chatMessage
  })
}

function estimateTokensForMessages(messages: ChatMessage[]): number {
  if (!messages.length) {
    return 0
  }
  const serialized = messages
    .map((message) => `${message.role}: ${message.content}`)
    .join('\n')
  return encode(serialized).length
}

function estimateTokensForText(text: string): number {
  if (!text) {
    return 0
  }
  return encode(text).length
}

function extractJsonSubstring(text: string): string | null {
  const trimmed = text.trim()
  const starts = [trimmed.indexOf('{'), trimmed.indexOf('[')].filter(
    (index) => index !== -1,
  )
  if (starts.length === 0) {
    return null
  }

  const start = Math.min(...starts)
  const openingChar = trimmed[start]
  const closingChar = openingChar === '{' ? '}' : ']'

  const stack: string[] = []
  let inString = false
  let escape = false

  for (let i = start; i < trimmed.length; i += 1) {
    const char = trimmed[i]

    if (inString) {
      if (escape) {
        escape = false
        continue
      }
      if (char === '\\') {
        escape = true
        continue
      }
      if (char === '"') {
        inString = false
      }
      continue
    }

    if (char === '"') {
      inString = true
      continue
    }

    if (char === '{' || char === '[') {
      stack.push(char === '{' ? '}' : ']')
      continue
    }

    if (char === '}' || char === ']') {
      const expected = stack.pop()
      if (char !== expected) {
        return null
      }
      if (stack.length === 0) {
        return trimmed.slice(start, i + 1)
      }
    }
  }

  return null
}

function parseStructuredOutput<T>(raw: string, schema: z.ZodType<T>): T {
  const candidates = new Set<string>()
  candidates.add(raw.trim())
  const extracted = extractJsonSubstring(raw)
  if (extracted) {
    candidates.add(extracted)
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate)
      return schema.parse(parsed)
    } catch {
      continue
    }
  }

  throw new Error('Structured response was not valid JSON')
}

export const promptAiSdkStream = async function* (
  options: PromptAiSdkStreamOptions,
): AsyncGenerator<StreamChunk, string | null> {
  if (
    !checkLiveUserInput(
      options.userId,
      options.userInputId,
      options.clientSessionId,
    )
  ) {
    logger.info(
      {
        userId: options.userId,
        userInputId: options.userInputId,
        liveUserInputId: getLiveUserInputIds(options.userId),
      },
      'Skipping stream due to canceled user input',
    )
    return null
  }

  const startTime = Date.now()
  const chatMessages = convertMessagesToChatMessages(
    options.messages,
    options.includeCacheControl,
  )
  const inputTokens = estimateTokensForMessages(chatMessages)
  const stopSequenceHandler = new StopSequenceHandler(options.stopSequences)

  let content = ''
  let stream: AsyncGenerator<ChatStreamChunk, void, unknown> | undefined
  let streamClosed = false

  try {
    stream = await chat(chatMessages, {
      stream: true,
      temperature: options.temperature,
      maxTokens: options.maxOutputTokens,
      stop: options.stopSequences,
      topP: options.topP,
      frequencyPenalty: options.frequencyPenalty,
      presencePenalty: options.presencePenalty,
      signal: options.signal,
    })

    const openrouterOptions = options.providerOptions?.openrouter
    const skipReasoning = Boolean(openrouterOptions?.reasoning?.exclude)

    for await (const chunk of stream) {
      if (chunk.type === 'reasoning') {
        if (!skipReasoning && chunk.text) {
          yield { type: 'reasoning', text: chunk.text }
        }
        continue
      }

      if (!chunk.text) {
        continue
      }

      if (!options.stopSequences?.length) {
        content += chunk.text
        yield { type: 'text', text: chunk.text }
        continue
      }

      const result = stopSequenceHandler.process(chunk.text)
      if (result.text) {
        content += result.text
        yield { type: 'text', text: result.text }
      }
      if (result.endOfStream) {
        streamClosed = true
        if (typeof stream.return === 'function') {
          await stream.return(undefined).catch(() => {})
        }
        break
      }
    }

    streamClosed = true
  } catch (error) {
    logger.error(
      {
        error: errorToObject(error),
        model: options.model,
      },
      'Error from OpenAI-compatible adapter',
    )
    const errorMessage = `Error from LLM adapter (model ${options.model}): ${buildArray([
      error instanceof Error ? error.message : String(error),
    ]).join('\n')}`
    yield { type: 'error', message: errorMessage }
    return null
  } finally {
    if (!streamClosed && stream && typeof stream.return === 'function') {
      await stream.return(undefined).catch(() => {})
    }
  }

  const flushed = stopSequenceHandler.flush()
  if (flushed) {
    content += flushed
    yield { type: 'text', text: flushed }
  }

  const outputTokens = estimateTokensForText(content)
  const messageId = generateCompactId()

  const creditsUsedPromise = saveMessage({
    messageId,
    userId: options.userId,
    clientSessionId: options.clientSessionId,
    fingerprintId: options.fingerprintId,
    userInputId: options.userInputId,
    model: options.model,
    request: options.messages,
    response: content,
    inputTokens,
    outputTokens,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
    finishedAt: new Date(),
    latencyMs: Date.now() - startTime,
    chargeUser: options.chargeUser ?? true,
    agentId: options.agentId,
  })

  if (options.onCostCalculated) {
    const creditsUsed = await creditsUsedPromise
    await options.onCostCalculated(creditsUsed)
  }

  return messageId
}

export const promptAiSdk = async function (
  options: PromptAiSdkOptions,
): Promise<string> {
  if (
    !checkLiveUserInput(
      options.userId,
      options.userInputId,
      options.clientSessionId,
    )
  ) {
    logger.info(
      {
        userId: options.userId,
        userInputId: options.userInputId,
        liveUserInputId: getLiveUserInputIds(options.userId),
      },
      'Skipping prompt due to canceled user input',
    )
    return ''
  }

  const startTime = Date.now()
  const chatMessages = convertMessagesToChatMessages(
    options.messages,
    options.includeCacheControl,
  )
  const inputTokens = estimateTokensForMessages(chatMessages)

  let content: string
  try {
    content = await chat(chatMessages, {
      temperature: options.temperature,
      maxTokens: options.maxTokens,
      stop: options.stopSequences,
      topP: options.topP,
      frequencyPenalty: options.frequencyPenalty,
      presencePenalty: options.presencePenalty,
    })
  } catch (error) {
    logger.error(
      {
        error: errorToObject(error),
        model: options.model,
      },
      'Error from OpenAI-compatible adapter',
    )
    throw error
  }

  const outputTokens = estimateTokensForText(content)

  const creditsUsedPromise = saveMessage({
    messageId: generateCompactId(),
    userId: options.userId,
    clientSessionId: options.clientSessionId,
    fingerprintId: options.fingerprintId,
    userInputId: options.userInputId,
    model: options.model,
    request: options.messages,
    response: content,
    inputTokens,
    outputTokens,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
    finishedAt: new Date(),
    latencyMs: Date.now() - startTime,
    chargeUser: options.chargeUser ?? true,
    agentId: options.agentId,
  })

  if (options.onCostCalculated) {
    const creditsUsed = await creditsUsedPromise
    await options.onCostCalculated(creditsUsed)
  }

  return content
}

export const promptAiSdkStructured = async function <T>(
  options: PromptAiSdkStructuredOptions<T>,
): Promise<T> {
  if (
    !checkLiveUserInput(
      options.userId,
      options.userInputId,
      options.clientSessionId,
    )
  ) {
    logger.info(
      {
        userId: options.userId,
        userInputId: options.userInputId,
        liveUserInputId: getLiveUserInputIds(options.userId),
      },
      'Skipping structured prompt due to canceled user input',
    )
    return {} as T
  }

  const startTime = Date.now()
  const chatMessages = convertMessagesToChatMessages(
    options.messages,
    options.includeCacheControl,
  )
  const inputTokens = estimateTokensForMessages(chatMessages)

  const chatPromise = chat(chatMessages, {
    temperature: options.temperature,
    maxTokens: options.maxTokens,
    stop: options.stopSequences,
    topP: options.topP,
    frequencyPenalty: options.frequencyPenalty,
    presencePenalty: options.presencePenalty,
  })

  let rawResponse: string
  try {
    rawResponse = await (options.timeout === undefined
      ? chatPromise
      : withTimeout(chatPromise, options.timeout))
  } catch (error) {
    logger.error(
      {
        error: errorToObject(error),
        model: options.model,
      },
      'Error from OpenAI-compatible adapter',
    )
    throw error
  }

  const parsed = parseStructuredOutput(rawResponse, options.schema)
  const serialized = JSON.stringify(parsed)
  const outputTokens = estimateTokensForText(serialized)

  const creditsUsedPromise = saveMessage({
    messageId: generateCompactId(),
    userId: options.userId,
    clientSessionId: options.clientSessionId,
    fingerprintId: options.fingerprintId,
    userInputId: options.userInputId,
    model: options.model,
    request: options.messages,
    response: serialized,
    inputTokens,
    outputTokens,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
    finishedAt: new Date(),
    latencyMs: Date.now() - startTime,
    chargeUser: options.chargeUser ?? true,
    agentId: options.agentId,
  })

  if (options.onCostCalculated) {
    const creditsUsed = await creditsUsedPromise
    await options.onCostCalculated(creditsUsed)
  }

  return parsed
}
