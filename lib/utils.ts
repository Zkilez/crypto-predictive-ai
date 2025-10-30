import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Formatea un timestamp a hora corta en UTC para evitar diferencias de locale/zonas
export function formatTimeUTC(timestamp: number, locale: string = "es-ES") {
  return new Date(timestamp).toLocaleTimeString(locale, {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  })
}

// Locale por código de moneda para formato consistente
export function localeForCurrency(code: string) {
  const map: Record<string, string> = {
    USD: "en-US",
    COP: "es-CO",
    EUR: "es-ES",
    MXN: "es-MX",
    BRL: "pt-BR",
  }
  return map[code] || "en-US"
}

// Formatea una cantidad en la moneda indicada, mostrando el código (USD, COP, etc.)
export function formatCurrencyByCode(
  value: number,
  currencyCode: string,
  options?: { maximumFractionDigits?: number; notation?: "compact" },
) {
  return new Intl.NumberFormat(localeForCurrency(currencyCode), {
    style: "currency",
    currency: currencyCode,
    currencyDisplay: "code",
    maximumFractionDigits:
      options?.maximumFractionDigits != null ? options.maximumFractionDigits : currencyCode === "COP" ? 0 : 2,
    notation: options?.notation,
  }).format(value)
}

// Versión compacta (ej. 1.2K, 3.4M)
export function formatCurrencyCompactByCode(value: number, currencyCode: string) {
  return formatCurrencyByCode(value, currencyCode, { notation: "compact" })
}
