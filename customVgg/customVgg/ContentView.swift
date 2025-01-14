//
//  ContentView.swift
//  customVgg
//
//  Created by jcd on 2025/1/9.
//
//
import Foundation
import CoreML	
import CoreVideo
import SwiftUI
import Accelerate

struct ContentView: View {
    @State private var showMessage = 0
    @State private var model: vgg_k3s2_norelu?
    @State private var model_a8w8: vgg_k3s2_norelu_a8w8?
    @State private var isModelExecuted = 0

    var body: some View {
        VStack {
            Button("Load fp16 Model") {
                loadModel(index: 1)
            }
            Button("Execute fp16 Model") {
                executeModel(index: 1)
            }
            Button("Load a8w8 Model") {
                loadModel(index: 2)
            }
            Button("Execute a8w8 Model") {
                executeModel(index: 2)
            }
            if showMessage == 1 {
                Text("fp16 Model loaded!")
            }
            if isModelExecuted == 1 {
                Text("fp16 Execution completed")
            }
            if showMessage == 2 {
                Text("a8w8 Model loaded!")
            }
            if isModelExecuted == 2 {
                Text("a8w8 Execution completed")
            }
        }
    }

    func loadModel(index: Int) {
        do {
            let configuration = MLModelConfiguration()
            if index == 1 {
                model = try vgg_k3s2_norelu(configuration: configuration)
            }else {
                model_a8w8 = try vgg_k3s2_norelu_a8w8(configuration: configuration)
            }
            //model = try vgg_k3s2(configuration: configuration)
            showMessage = index
        } catch {
            print("Error loading or using the model: \(error)")
        }
    }

    func executeModel(index: Int) {
        guard let model = model else {
            print("Model not loaded")
            return
        }

        do {
//            let pixelBuffer = createPixelBuffer()
            let width = 448
            let height = 448
            let depth = 64
            let batch = 1

            let multiArrayBuffer = createMLMultiArray(width: width, height: height, depth: depth, batchcount: batch)
            if let buffer = multiArrayBuffer {
                
                if index == 1 {
                    let input = vgg_k3s2_noreluInput(input: buffer)
                    for i in 0..<10 {
                        _ = try model.prediction(input: input)
                        print("fp16 model exect for \(i)th times")
                    }
                } else {
                    let input = vgg_k3s2_norelu_a8w8Input(input: buffer)
                    for i in 0..<10 {
                        _ = try model_a8w8?.prediction(input: input)
                        print("a8w8 model exect for \(i)th times")
                    }
                }

                isModelExecuted = index
                // 后续处理
            } else {
                // 处理像素缓冲区创建失败的情况
            }
        } catch {
            print("Error executing the model: \(error)")
        }
    }

    func createPixelBuffer() -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, 224, 224, kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pixelBuffer)
        if status != kCVReturnSuccess {
            print("Error creating pixel buffer")
            return nil
        }
        return pixelBuffer
    }


    // 创建指定形状的 MultiArray
    func createMLMultiArray(width: Int, height: Int, depth: Int, batchcount: Int) -> MLMultiArray? {
        let dataType = MLMultiArrayDataType.float16
        let shape = [NSNumber(value: batchcount), NSNumber(value: depth), NSNumber(value: height), NSNumber(value: width)]
        return try? MLMultiArray(shape: shape, dataType: dataType)
    }
}

#Preview {
    ContentView()
//    ViewController()
}
