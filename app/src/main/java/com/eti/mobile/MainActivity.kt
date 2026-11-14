package com.eti.mobile

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.webkit.SafeBrowsingResponse
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var webView: WebView
    private val trustedHost = "eti-api-production.up.railway.app"
    private val startUrl = "https://eti-api-production.up.railway.app/"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.rgb(12, 18, 32)
        window.navigationBarColor = Color.rgb(12, 18, 32)
        openResearchApp()
    }

    private fun openResearchApp() {
        webView = WebView(this)
        webView.setBackgroundColor(Color.rgb(12, 18, 32))
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = false
            allowFileAccess = false
            allowContentAccess = false
            javaScriptCanOpenWindowsAutomatically = false
            setSupportMultipleWindows(false)
            setSupportZoom(false)
            builtInZoomControls = false
            displayZoomControls = false
            mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
            safeBrowsingEnabled = true
            loadsImagesAutomatically = true
        }
        webView.webChromeClient = WebChromeClient()
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val uri = request.url
                if (uri.scheme == "https" && uri.host == trustedHost) return false
                if (uri.scheme == "https" || uri.scheme == "http") {
                    startActivity(Intent(Intent.ACTION_VIEW, uri))
                }
                return true
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: android.webkit.WebResourceError
            ) {
                if (request.isForMainFrame) showOfflineMessage()
            }

            override fun onSafeBrowsingHit(
                view: WebView,
                request: WebResourceRequest,
                threatType: Int,
                callback: SafeBrowsingResponse
            ) {
                callback.backToSafety(true)
            }
        }
        setContentView(webView)
        webView.loadUrl(startUrl)
    }

    private fun showOfflineMessage() {
        val message = TextView(this).apply {
            text = "ETI could not connect. Check your internet connection and try again."
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.rgb(12, 18, 32))
            gravity = android.view.Gravity.CENTER
            setPadding(32, 32, 32, 32)
            setOnClickListener { webView.reload() }
        }
        setContentView(message)
    }

    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        if (::webView.isInitialized && webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        if (::webView.isInitialized) {
            webView.stopLoading()
            webView.destroy()
        }
        super.onDestroy()
    }
}
