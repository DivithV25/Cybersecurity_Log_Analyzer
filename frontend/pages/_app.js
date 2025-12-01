import React from 'react'
import '../styles/global.css'

// Ensure we export a valid React component for Next.js
export default function App({ Component, pageProps }) {
  return <Component {...pageProps} />
}
